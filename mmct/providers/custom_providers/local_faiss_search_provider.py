import os
import json
import threading
import uuid
from typing import Dict, Any, List, Optional
import asyncio

import numpy as np
import faiss
from types import SimpleNamespace

from loguru import logger
from mmct.providers.base import SearchProvider
from mmct.utils.error_handler import ProviderException, handle_exceptions, convert_exceptions


class LocalFaissSearchProvider(SearchProvider):
    """Local FAISS-backed search provider.

    This provider:
    - Stores FAISS indexes on-disk under `index_path` (config) or mmct_faiss_indices
    - Persists metadata JSON per index mapping document IDs to internal numeric IDs
    - Exposes async interface by running blocking FAISS calls in background threads
    - Supports both regular embeddings and CLIP embeddings
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.base_path = self.config.get("index_path", "mmct_faiss_indices")
        os.makedirs(self.base_path, exist_ok=True)

        # Runtime caches for indexes, metadata, and thread locks
        self._indexes: Dict[str, faiss.Index] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}

    # ============================================================================
    # Helper Methods - File paths and synchronization
    # ============================================================================
    
    def _index_file(self, index_name: str) -> str:
        """Get the file path for the FAISS index."""
        return os.path.join(self.base_path, f"{index_name}.index")

    def _meta_file(self, index_name: str) -> str:
        """Get the file path for the metadata JSON."""
        return os.path.join(self.base_path, f"{index_name}.meta.json")

    def _ensure_lock(self, index_name: str) -> None:
        """Ensure a lock exists for the given index."""
        if index_name not in self._locks:
            self._locks[index_name] = threading.Lock()

    # ============================================================================
    # Helper Methods - Index and metadata persistence
    # ============================================================================
    
    def _load_index_sync(self, index_name: str) -> None:
        """Blocking load of index and metadata if present."""
        self._ensure_lock(index_name)
        with self._locks[index_name]:
            if index_name in self._indexes:
                return

            meta_path = self._meta_file(index_name)
            index_path = self._index_file(index_name)

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception as e:
                    # Corrupt or partial JSON — move the broken file aside and start fresh
                    try:
                        logger.exception(f"Failed to load meta JSON for index '{index_name}': {e}")
                        corrupt_path = meta_path + ".corrupt"
                        os.replace(meta_path, corrupt_path)
                        logger.warning(f"Moved corrupt meta file to {corrupt_path} and reinitializing meta")
                    except Exception:
                        logger.exception("Failed to move corrupt meta file")
                    meta = {"next_id": 1, "docid_to_id": {}, "id_to_docid": {}, "docs": {}, "dim": None}
            else:
                meta = {"next_id": 1, "docid_to_id": {}, "id_to_docid": {}, "docs": {}, "dim": None}

            if os.path.exists(index_path):
                try:
                    idx = faiss.read_index(index_path)
                except Exception:
                    logger.exception("Failed to read FAISS index file; initializing new index")
                    idx = None
            else:
                idx = None

            self._meta[index_name] = meta
            self._indexes[index_name] = idx

    def _save_index_sync(self, index_name: str) -> None:
        """Blocking save of index and metadata."""
        self._ensure_lock(index_name)
        with self._locks[index_name]:
            meta = self._meta.get(index_name)
            idx = self._indexes.get(index_name)
            if meta is None:
                return
            # Atomically write metadata to avoid partial/corrupt files (write to temp then replace)
            meta_path = self._meta_file(index_name)
            tmp_meta_path = meta_path + ".tmp"
            try:
                with open(tmp_meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, default=str, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_meta_path, meta_path)
            except Exception:
                logger.exception("Failed to persist FAISS meta JSON atomically")
                # best-effort: try to remove tmp file
                try:
                    if os.path.exists(tmp_meta_path):
                        os.remove(tmp_meta_path)
                except Exception:
                    pass

            # Persist FAISS index atomically by writing to a tmp file then replacing
            if idx is not None:
                index_path = self._index_file(index_name)
                tmp_index_path = index_path + ".tmp"
                try:
                    faiss.write_index(idx, tmp_index_path)
                    os.replace(tmp_index_path, index_path)
                except Exception:
                    logger.exception("Failed to persist FAISS index atomically")
                    try:
                        if os.path.exists(tmp_index_path):
                            os.remove(tmp_index_path)
                    except Exception:
                        pass

    # ============================================================================
    # Helper Methods - Index initialization and document processing
    # ============================================================================
    
    def _initialize_index_if_needed(self, meta: Dict[str, Any], embeddings: List[float], index_name: str) -> faiss.Index:
        """
        Initialize FAISS index if it doesn't exist yet.
        
        Args:
            meta: Metadata dictionary for the index
            embeddings: Sample embeddings to determine dimensionality
            index_name: Name of the index
            
        Returns:
            The initialized or existing FAISS index
        """
        idx = self._indexes[index_name]
        
        # Get dimension from embeddings
        embedding_dim = len(embeddings)
        
        # Initialize index if dimension not set or if index doesn't exist
        if meta.get("dim") is None:
            dim = embedding_dim
            meta["dim"] = dim
            base_index = faiss.IndexFlatL2(dim)
            idx = faiss.IndexIDMap(base_index)
            self._indexes[index_name] = idx
        elif idx is None:
            # Index metadata exists but index object doesn't - recreate it
            dim = meta["dim"]
            base_index = faiss.IndexFlatL2(dim)
            idx = faiss.IndexIDMap(base_index)
            self._indexes[index_name] = idx
        else:
            # Verify dimension matches
            existing_dim = meta.get("dim")
            if existing_dim != embedding_dim:
                raise ProviderException(
                    f"Dimension mismatch for index '{index_name}': "
                    f"existing dimension is {existing_dim}, but document has {embedding_dim}. "
                    f"Cannot add documents with different dimensions to the same index."
                )
            
        # Ensure idx is IDMap for ID-based operations
        if not isinstance(idx, faiss.IndexIDMap):
            base = idx if idx is not None else faiss.IndexFlatL2(meta["dim"])
            idx = faiss.IndexIDMap(base)
            self._indexes[index_name] = idx
            
        return idx

    def _add_document_to_index(
        self, 
        idx: faiss.Index, 
        meta: Dict[str, Any], 
        docid: str, 
        embeddings: List[float], 
        document: Dict[str, Any]
    ) -> None:
        """
        Add or update a document in the FAISS index.
        
        Args:
            idx: FAISS index to add to
            meta: Metadata dictionary
            docid: Document identifier
            embeddings: Document embeddings
            document: Full document data
        """
        # Assign numeric ID
        if docid in meta["docid_to_id"]:
            # Update existing document - remove old vector first
            numeric_id = int(meta["docid_to_id"][docid])
            try:
                idx.remove_ids(np.array([numeric_id], dtype=np.int64))
            except Exception:
                # Some FAISS versions may not support remove_ids; ignore
                pass
        else:
            # New document - get next available ID
            numeric_id = meta.get("next_id", 1)
            meta["next_id"] = numeric_id + 1

        # Add vector to index
        vec = np.array(embeddings, dtype=np.float32).reshape(1, -1)
        idx.add_with_ids(vec, np.array([numeric_id], dtype=np.int64))

        # Update metadata mappings
        meta["docid_to_id"][docid] = numeric_id
        meta["id_to_docid"][str(numeric_id)] = docid
        meta["docs"][docid] = document

    def _perform_vector_search(
        self, 
        idx: faiss.Index, 
        meta: Dict[str, Any], 
        embedding: List[float], 
        top: int,
        index_name: str
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using FAISS.
        
        Args:
            idx: FAISS index to search
            meta: Metadata dictionary
            embedding: Query embedding
            top: Number of results to return
            index_name: Name of the index (for logging)
            
        Returns:
            List of search results with id, score, and document
        """
        if idx is None:
            return []
            
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        try:
            D, I = idx.search(vec, top)
        except Exception as e:
            logger.error(f"FAISS search failed for index '{index_name}': {e}")
            return []
            
        ids = I[0].tolist()
        distances = D[0].tolist()
        logger.debug(f"FAISS search returned {len(ids)} results for index '{index_name}'")
        
        results = []
        for nid, dist in zip(ids, distances):
            if int(nid) == -1:  # Invalid ID
                continue
            docid = meta["id_to_docid"].get(str(int(nid)))
            if not docid:
                continue
            doc = meta["docs"].get(docid)
            if doc:
                results.append({"id": docid, "score": float(dist), "document": doc})
                
        return results

    def _perform_text_search(self, meta: Dict[str, Any], query: str, text_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Perform fallback text-based search when embeddings not available.
        
        Args:
            meta: Metadata dictionary
            query: Search query text
            text_fields: List of document fields to search in
            
        Returns:
            List of matching documents
        """
        results = []
        text = query.lower() if query else ""
        
        for docid, doc in meta.get("docs", {}).items():
            combined_text = []
            for field in text_fields:
                if field in doc and doc[field]:
                    combined_text.append(str(doc[field]))
            combined = " ".join(combined_text).lower()
            
            if text and text in combined:
                results.append({"id": docid, "score": 1.0, "document": doc})
                
        return results

    # ============================================================================
    # Public API Methods - Index management
    # ============================================================================
    
    async def create_index(self, index_name: str, index_schema: Any) -> bool:
        """
        Create a new index.
        
        Args:
            index_name: Name of the index
            index_schema: Can be:
                - String: "chapter" (dim=1536) or "keyframe" (dim=512)
                - Dict: {"type": "chapter", "dim": 1536} or {"type": "keyframe", "dim": 512} or {"dim": 512}
                - None: dimension will be inferred from first document
                
        Returns:
            bool: True if successful
        """
        try:
            await asyncio.to_thread(self._load_index_sync, index_name)
            meta = self._meta[index_name]
            if meta.get("dim") is not None:
                return True

            # Handle different schema formats and set default dimensions
            dim = None
            if isinstance(index_schema, str):
                # String indicator: use type-specific defaults
                if index_schema == "chapter":
                    dim = 1536  # text-embedding-ada-002 default
                elif index_schema == "keyframe":
                    dim = 512   # CLIP ViT-B/32 default
            elif isinstance(index_schema, dict):
                # Dict format: extract dim or use type-based default
                dim = index_schema.get("dim")
                if dim is None:
                    schema_type = index_schema.get("type")
                    if schema_type == "chapter":
                        dim = 1536
                    elif schema_type == "keyframe":
                        dim = 512
            # For None or unrecognized types, dim remains None and will be inferred from first document
            
            meta["dim"] = dim
            await asyncio.to_thread(self._save_index_sync, index_name)
            return True
        except Exception as e:
            logger.error(f"Failed to create FAISS index '{index_name}': {e}")
            raise ProviderException(f"Failed to create FAISS index '{index_name}': {e}")

    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        await asyncio.to_thread(self._load_index_sync, index_name)
        meta = self._meta.get(index_name, {})
        return bool(meta and (meta.get("dim") is not None or os.path.exists(self._index_file(index_name))))

    # ============================================================================
    # Public API Methods - Document operations
    # ============================================================================
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_document(self, document: Dict, index_name: str = None) -> bool:
        """
        Index a single document or a list of documents.

        Args:
            document: Document or list of documents to index
            index_name: Name of the index
            
        Returns:
            bool: True if successful
            
        Note:
            Documents must include an 'embeddings' field (list[float]) and 
            an 'id' or 'hash_video_id'.
        """
        try:
            if document is None:
                raise ProviderException("Document is empty")

            # Handle batch indexing
            if isinstance(document, list):
                results = []
                for doc in document:
                    results.append(await self.index_document(doc, index_name=index_name))
                return all(results)

            # Single document indexing
            await asyncio.to_thread(self._load_index_sync, index_name)
            meta = self._meta[index_name]

            embeddings = document.get("embeddings")
            if not embeddings:
                raise ProviderException("Document missing 'embeddings' field")

            docid = document.get("id") or document.get("hash_video_id") or str(uuid.uuid4())

            # Initialize index if needed and get the index
            idx = self._initialize_index_if_needed(meta, embeddings, index_name)

            # Add document to index
            self._add_document_to_index(idx, meta, docid, embeddings, document)

            # Persist changes
            await asyncio.to_thread(self._save_index_sync, index_name)
            return True
        except Exception as e:
            logger.exception(f"Local FAISS indexing failed: {e}")
            raise ProviderException(f"Local FAISS indexing failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_documents(self, documents: List[Dict], index_name: str = None) -> Dict[str, Any]:
        if not documents:
            return {"success": False, "count": 0, "message": "No documents provided"}

        success = 0
        for doc in documents:
            ok = await self.index_document(doc, index_name=index_name)
            if ok:
                success += 1

        return {"success": True, "count": success}

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search(self, query: str, index_name: str = None, **kwargs) -> List[Dict]:
        """
        Perform vector search using provided embedding in kwargs.

        Args:
            query: Search query string (used for fallback text search)
            index_name: Name of the index to search
            **kwargs: Additional parameters:
                - embedding: Query embedding for vector search
                - top: Number of results to return (default: 5)
                
        Returns:
            List of search results
            
        Note:
            If embedding not provided, falls back to substring match on text fields.
        """
        try:
            embedding = kwargs.get("embedding")
            top = kwargs.get("top", 5)

            await asyncio.to_thread(self._load_index_sync, index_name)
            meta = self._meta[index_name]
            idx = self._indexes[index_name]

            # Log search state for debugging
            num_docs = len(meta.get("docs", {})) if meta else 0
            logger.debug(
                f"Search on index '{index_name}': dim={meta.get('dim') if meta else None}, "
                f"docs={num_docs}, has_index={idx is not None}, has_embedding={embedding is not None}"
            )

            # Vector search if embedding provided
            if embedding is not None and idx is not None:
                return self._perform_vector_search(idx, meta, embedding, top, index_name)

            # Fallback: text-based substring search
            text_fields = ["detailed_summary", "text_from_scene", "chapter_transcript"]
            return self._perform_text_search(meta, query, text_fields)
            
        except Exception as e:
            logger.error(f"Local FAISS search failed: {e}")
            raise ProviderException(f"Local FAISS search failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_document(self, doc_id: str, index_name: str = None) -> bool:
        try:
            await asyncio.to_thread(self._load_index_sync, index_name)
            meta = self._meta[index_name]
            idx = self._indexes[index_name]

            if doc_id not in meta["docid_to_id"]:
                return False

            numeric_id = int(meta["docid_to_id"].pop(doc_id))
            meta["id_to_docid"].pop(str(numeric_id), None)
            meta["docs"].pop(doc_id, None)

            try:
                idx.remove_ids(np.array([numeric_id], dtype=np.int64))
            except Exception:
                logger.warning("FAISS remove_ids not supported or failed; index may still contain vector")

            await asyncio.to_thread(self._save_index_sync, index_name)
            return True
        except Exception as e:
            logger.error(f"Local FAISS delete failed: {e}")
            raise ProviderException(f"Local FAISS delete failed: {e}")

    async def delete_index(self, index_name: str) -> bool:
        try:
            await asyncio.to_thread(self._load_index_sync, index_name)
            # remove files
            idx_path = self._index_file(index_name)
            meta_path = self._meta_file(index_name)
            try:
                if os.path.exists(idx_path):
                    os.remove(idx_path)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            except Exception:
                logger.exception("Failed to delete index files")

            self._indexes.pop(index_name, None)
            self._meta.pop(index_name, None)
            return True
        except Exception as e:
            logger.error(f"Local FAISS delete index failed: {e}")
            raise ProviderException(f"Local FAISS delete index failed: {e}")

    async def check_is_document_exist(self, hash_id: str, index_name: str = None) -> bool:
        await asyncio.to_thread(self._load_index_sync, index_name)
        meta = self._meta.get(index_name, {})
        return hash_id in meta.get("docid_to_id", {})

    # ============================================================================
    # Cleanup
    # ============================================================================
    
    async def close(self) -> None:
        """Persist all indexes to disk before closing."""
        for index_name in list(self._meta.keys()):
            await asyncio.to_thread(self._save_index_sync, index_name)

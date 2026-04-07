"""Local FAISS-based object vector database provider."""

import os
import json
import asyncio
import threading
import uuid
from typing import Dict, Any, List, Optional

import numpy as np

from loguru import logger
from mmct.providers.base.object_vector_db_provider import BaseObjectVectorDBProvider


class LocalFAISSObjectProvider(BaseObjectVectorDBProvider):
    """Local FAISS-backed object vector database provider.
    
    Stores object documents with vector embeddings using FAISS for similarity search.
    Persists indexes and metadata to disk for durability.
    """
    
    def __init__(
        self,
        index_name: str = "objects",
        index_path: Optional[str] = "mmct_faiss_indices"
    ):
        """Initialize the FAISS object provider.
        
        Args:
            index_name: Name of the index.
            index_path: Directory to store index files.
        """
        super().__init__(index_name)
        self.base_path = index_path
        os.makedirs(self.base_path, exist_ok=True)
        
        self._index = None
        self._meta: Dict[str, Any] = {
            "next_id": 1,
            "docid_to_id": {},
            "id_to_docid": {},
            "docs": {},
            "dim": None
        }
        self._lock = threading.Lock()
        self._loaded = False
    
    def _index_file(self) -> str:
        """Get path to FAISS index file."""
        return os.path.join(self.base_path, f"{self.index_name}_objects.index")
    
    def _meta_file(self) -> str:
        """Get path to metadata JSON file."""
        return os.path.join(self.base_path, f"{self.index_name}_objects.meta.json")
    
    def _load_sync(self) -> None:
        """Load index and metadata from disk."""
        with self._lock:
            if self._loaded:
                return
            
            meta_path = self._meta_file()
            index_path = self._index_file()
            
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self._meta = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load meta file: {e}")
                    self._meta = {
                        "next_id": 1, "docid_to_id": {},
                        "id_to_docid": {}, "docs": {}, "dim": None
                    }
            
            if os.path.exists(index_path):
                try:
                    import faiss
                    self._index = faiss.read_index(index_path)
                except Exception as e:
                    logger.warning(f"Failed to load FAISS index: {e}")
                    self._index = None
            
            self._loaded = True
    
    def _save_sync(self) -> None:
        """Save index and metadata to disk."""
        with self._lock:
            meta_path = self._meta_file()
            tmp_meta_path = meta_path + ".tmp"
            
            try:
                with open(tmp_meta_path, "w", encoding="utf-8") as f:
                    json.dump(self._meta, f, default=str, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_meta_path, meta_path)
            except Exception as e:
                logger.error(f"Failed to save meta file: {e}")
                if os.path.exists(tmp_meta_path):
                    os.remove(tmp_meta_path)
            
            if self._index is not None:
                import faiss
                index_path = self._index_file()
                tmp_index_path = index_path + ".tmp"
                try:
                    faiss.write_index(self._index, tmp_index_path)
                    os.replace(tmp_index_path, index_path)
                except Exception as e:
                    logger.error(f"Failed to save FAISS index: {e}")
                    if os.path.exists(tmp_index_path):
                        os.remove(tmp_index_path)
    
    def _ensure_index(self, dim: int) -> None:
        """Ensure FAISS index exists with given dimension."""
        import faiss
        if self._index is None:
            self._meta["dim"] = dim
            base_index = faiss.IndexFlatL2(dim)
            self._index = faiss.IndexIDMap(base_index)
    
    async def search(
        self,
        query: str,
        video_id: Optional[str] = None,
        object_type: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search objects by text query with metadata filtering."""
        await asyncio.to_thread(self._load_sync)
        
        results = []
        for doc_id, doc in self._meta["docs"].items():
            if video_id and doc.get("video_id") != video_id:
                continue
            if object_type and doc.get("object_type") != object_type:
                continue
            
            name = doc.get("name", "")
            if query.lower() in name.lower():
                results.append({
                    "id": doc_id,
                    "score": 1.0,
                    "document": doc
                })
        
        return results[:limit]
    
    async def search_by_vector(
        self,
        query_vector: List[float],
        video_id: Optional[str] = None,
        object_type: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search objects by vector similarity."""
        await asyncio.to_thread(self._load_sync)
        
        if self._index is None or self._index.ntotal == 0:
            return []
        
        vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        
        search_limit = min(limit * 3, self._index.ntotal)
        D, I = self._index.search(vec, search_limit)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            
            doc_id = self._meta["id_to_docid"].get(str(int(idx)))
            if not doc_id:
                continue
            
            doc = self._meta["docs"].get(doc_id)
            if not doc:
                continue
            
            if video_id and doc.get("video_id") != video_id:
                continue
            if object_type and doc.get("object_type") != object_type:
                continue
            
            similarity = 1.0 / (1.0 + float(dist))
            results.append({
                "id": doc_id,
                "score": similarity,
                "document": doc
            })
            
            if len(results) >= limit:
                break
        
        return results
    
    async def search_similar_objects(
        self,
        object_id: str,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find objects similar to a given object."""
        await asyncio.to_thread(self._load_sync)
        
        doc = self._meta["docs"].get(object_id)
        if not doc:
            return []
        
        embedding = doc.get("embedding_vector")
        if not embedding:
            return []
        
        results = await self.search_by_vector(embedding, limit=limit + 1)
        return [r for r in results if r["id"] != object_id][:limit]
    
    async def index_document(
        self,
        document: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Index a single object document."""
        await asyncio.to_thread(self._load_sync)
        
        doc_id = document.get("id") or str(uuid.uuid4())
        embedding = document.get("embedding_vector")
        
        with self._lock:
            if embedding:
                import faiss
                self._ensure_index(len(embedding))
                
                if doc_id in self._meta["docid_to_id"]:
                    old_id = int(self._meta["docid_to_id"][doc_id])
                    try:
                        self._index.remove_ids(np.array([old_id], dtype=np.int64))
                    except Exception:
                        pass
                
                numeric_id = self._meta["next_id"]
                self._meta["next_id"] += 1
                
                vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
                self._index.add_with_ids(vec, np.array([numeric_id], dtype=np.int64))
                
                self._meta["docid_to_id"][doc_id] = numeric_id
                self._meta["id_to_docid"][str(numeric_id)] = doc_id
            
            self._meta["docs"][doc_id] = document
        
        await asyncio.to_thread(self._save_sync)
        return True
    
    async def create_index(
        self,
        schema: Optional[Any] = None,
        **kwargs
    ) -> bool:
        """Create the search index."""
        await asyncio.to_thread(self._load_sync)
        
        dim = None
        if isinstance(schema, dict):
            dim = schema.get("dim")
        elif isinstance(schema, int):
            dim = schema
        
        if dim:
            import faiss
            with self._lock:
                if self._index is None:
                    self._meta["dim"] = dim
                    base_index = faiss.IndexFlatL2(dim)
                    self._index = faiss.IndexIDMap(base_index)
        
        await asyncio.to_thread(self._save_sync)
        return True
    
    async def delete_document(
        self,
        document_id: str,
        **kwargs
    ) -> bool:
        """Delete a document from the index."""
        await asyncio.to_thread(self._load_sync)
        
        with self._lock:
            if document_id not in self._meta["docs"]:
                return False
            
            if document_id in self._meta["docid_to_id"]:
                numeric_id = int(self._meta["docid_to_id"][document_id])
                try:
                    import faiss
                    self._index.remove_ids(np.array([numeric_id], dtype=np.int64))
                except Exception:
                    pass
                
                del self._meta["docid_to_id"][document_id]
                self._meta["id_to_docid"].pop(str(numeric_id), None)
            
            del self._meta["docs"][document_id]
        
        await asyncio.to_thread(self._save_sync)
        return True
    
    async def index_exists(self) -> bool:
        """Check if the index exists."""
        await asyncio.to_thread(self._load_sync)
        return self._meta.get("dim") is not None or os.path.exists(self._index_file())
    
    async def delete_index(self) -> bool:
        """Delete the search index."""
        with self._lock:
            idx_path = self._index_file()
            meta_path = self._meta_file()
            
            try:
                if os.path.exists(idx_path):
                    os.remove(idx_path)
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            except Exception as e:
                logger.error(f"Failed to delete index files: {e}")
                return False
            
            self._index = None
            self._meta = {
                "next_id": 1, "docid_to_id": {},
                "id_to_docid": {}, "docs": {}, "dim": None
            }
            self._loaded = False
        
        return True
    
    async def close(self) -> None:
        """Save and close the provider."""
        await asyncio.to_thread(self._save_sync)
        logger.info(f"LocalFAISSObjectProvider closed for index: {self.index_name}")

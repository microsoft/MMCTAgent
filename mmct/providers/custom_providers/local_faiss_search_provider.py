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

    - Stores FAISS indexes on-disk under `index_path` (config) or mmct_faiss_indices
    - Persists a small metadata JSON per index that maps document ids to internal int ids
    - Exposes the same async interface expected by the system (methods run blocking FAISS calls
      in a background thread to avoid blocking the event loop)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.base_path = self.config.get("index_path", "mmct_faiss_indices")
        os.makedirs(self.base_path, exist_ok=True)

        # runtime caches
        self._indexes: Dict[str, faiss.Index] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}

    def _index_file(self, index_name: str) -> str:
        return os.path.join(self.base_path, f"{index_name}.index")

    def _meta_file(self, index_name: str) -> str:
        return os.path.join(self.base_path, f"{index_name}.meta.json")

    def _ensure_lock(self, index_name: str):
        if index_name not in self._locks:
            self._locks[index_name] = threading.Lock()

    def _load_index_sync(self, index_name: str):
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

    def _save_index_sync(self, index_name: str):
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

    async def create_index(self, index_name: str, index_schema: Any) -> bool:
        """Create a new index. If index_schema contains `dim`, use it to create the index immediately."""
        try:
            await asyncio.to_thread(self._load_index_sync, index_name)
            meta = self._meta[index_name]
            if meta.get("dim") is not None:
                return False

            dim = None
            if isinstance(index_schema, dict):
                dim = index_schema.get("dim")

            meta["dim"] = dim
            await asyncio.to_thread(self._save_index_sync, index_name)
            return True
        except Exception as e:
            logger.error(f"Failed to create FAISS index '{index_name}': {e}")
            raise ProviderException(f"Failed to create FAISS index '{index_name}': {e}")

    def _initialize_client(self):
        """Compatibility shim used by code that expects a `client` with `_index_name` attribute.

        Returns a simple proxy object with `_index_name` set from config (or 'default').
        """
        index_name = self.config.get("index_name") or "default"
        client = SimpleNamespace()
        client._index_name = index_name
        # expose a no-op close() for compatibility
        async def _close():
            return None
        client.close = _close
        return client

    async def index_exists(self, index_name: str) -> bool:
        await asyncio.to_thread(self._load_index_sync, index_name)
        meta = self._meta.get(index_name, {})
        return bool(meta and (meta.get("dim") is not None or os.path.exists(self._index_file(index_name))))

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_document(self, document: Dict, index_name: str = None) -> bool:
        """Index a single document or a list of documents.

        Documents must include an `embeddings` field (list[float]) and an `id` or `hash_video_id`.
        """
        try:
            if document is None:
                raise ProviderException("Document is empty")

            if isinstance(document, list):
                # batch
                results = []
                for doc in document:
                    results.append(await self.index_document(doc, index_name=index_name))
                return all(results)

            # single doc
            await asyncio.to_thread(self._load_index_sync, index_name)
            meta = self._meta[index_name]
            idx = self._indexes[index_name]

            embeddings = document.get("embeddings")
            if not embeddings:
                raise ProviderException("Document missing 'embeddings' field")

            docid = document.get("id") or document.get("hash_video_id") or str(uuid.uuid4())

            # initialize index if needed
            if meta.get("dim") is None:
                dim = len(embeddings)
                meta["dim"] = dim
                # create a flat L2 index and wrap with ID map
                base_index = faiss.IndexFlatL2(dim)
                idx = faiss.IndexIDMap(base_index)
                self._indexes[index_name] = idx

            # ensure idx is IDMap
            if not isinstance(idx, faiss.IndexIDMap):
                # wrap
                base = idx if idx is not None else faiss.IndexFlatL2(meta["dim"])
                idx = faiss.IndexIDMap(base)
                self._indexes[index_name] = idx

            # assign numeric id
            if docid in meta["docid_to_id"]:
                # update — remove existing id then re-add
                numeric_id = int(meta["docid_to_id"][docid])
                try:
                    idx.remove_ids(np.array([numeric_id], dtype=np.int64))
                except Exception:
                    # some FAISS versions may not support remove_ids for this wrapper; ignore
                    pass
            else:
                numeric_id = meta.get("next_id", 1)
                meta["next_id"] = numeric_id + 1

            vec = np.array(embeddings, dtype=np.float32).reshape(1, -1)
            # add with ids
            idx.add_with_ids(vec, np.array([numeric_id], dtype=np.int64))

            # persist doc metadata
            meta["docid_to_id"][docid] = numeric_id
            meta["id_to_docid"][str(numeric_id)] = docid
            meta["docs"][docid] = document

            # save
            await asyncio.to_thread(self._save_index_sync, index_name)
            return True
        except Exception as e:
            logger.error(f"Local FAISS indexing failed: {e}")
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
        """Perform vector search using provided `embedding` in kwargs.

        If `embedding` not provided, falls back to simple substring match on stored documents' 'detailed_summary' or 'text_from_scene'.
        """
        try:
            embedding = kwargs.get("embedding")
            top = kwargs.get("top", 5)

            await asyncio.to_thread(self._load_index_sync, index_name)
            meta = self._meta[index_name]
            idx = self._indexes[index_name]

            # Debug: reveal meta/index state for troubleshooting
            try:
                num_docs = len(meta.get("docs", {})) if meta else 0
            except Exception:
                num_docs = 0
            print(f"[LocalFaissSearchProvider] search called index_name={index_name}, meta_dim={meta.get('dim') if meta else None}, num_docs={num_docs}, idx_present={idx is not None}")

            results = []
            if embedding is not None and idx is not None:
                vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
                try:
                    D, I = idx.search(vec, top)
                except Exception as e:
                    print(f"[LocalFaissSearchProvider] faiss.index.search failed: {e}")
                    return []
                ids = I[0].tolist()
                distances = D[0].tolist()
                print(f"[LocalFaissSearchProvider] faiss returned ids={ids[:10]} distances={distances[:10]}")
                for nid, dist in zip(ids, distances):
                    if int(nid) == -1:
                        continue
                    docid = meta["id_to_docid"].get(str(int(nid)))
                    if not docid:
                        continue
                    doc = meta["docs"].get(docid)
                    results.append({"id": docid, "score": float(dist), "document": doc})
                return results

            # fallback: substring search
            text = query.lower() if query else ""
            for docid, doc in meta.get("docs", {}).items():
                text_fields = []
                for fld in ("detailed_summary", "text_from_scene", "chapter_transcript"):
                    if fld in doc and doc[fld]:
                        text_fields.append(str(doc[fld]))
                combined = " ".join(text_fields).lower()
                if text and text in combined:
                    results.append({"id": docid, "score": 1.0, "document": doc})

            return results
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

    async def close(self):
        # nothing to close for FAISS in-memory objects beyond persisting
        for index_name in list(self._meta.keys()):
            await asyncio.to_thread(self._save_index_sync, index_name)

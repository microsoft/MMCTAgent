"""FAISS-based provider for chapter group vector database with order support."""

from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from mmct.providers.base.group_vector_db_provider import BaseGroupVectorDBProvider

logger = logging.getLogger(__name__)


class LocalFAISSGroupProvider(BaseGroupVectorDBProvider):
    """Local FAISS-based provider for chapter group storage and retrieval.
    
    Stores groups with order metadata for efficient sequential navigation.
    Supports filtering by video_id and ordering by group order.
    
    Attributes:
        index_path: Path to save/load FAISS index.
        dimension: Embedding vector dimension.
    """
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        dimension: int = 1536,
    ):
        """Initialize the FAISS group provider.
        
        Args:
            index_path: Path for index persistence.
            dimension: Embedding dimension (default: 1536 for Azure OpenAI).
        """
        if faiss is None:
            raise ImportError("faiss-cpu or faiss-gpu is required")
        
        self.index_path = Path(index_path) if index_path else None
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
        
        # Video-specific order index for fast lookups
        self._video_order_index: Dict[str, Dict[int, int]] = {}  # video_id -> {order -> doc_idx}
    
    async def create_index(
        self,
        dimension: int,
        index_name: str = "groups",
    ) -> bool:
        """Create a new FAISS index for groups."""
        try:
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
            self.documents = []
            self._id_to_idx = {}
            self._video_order_index = {}
            logger.info(f"Created FAISS group index with dimension {dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    async def index_document(
        self,
        document: Dict[str, Any],
    ) -> bool:
        """Index a group document with order metadata."""
        try:
            if self.index is None:
                await self.create_index(self.dimension)
            
            doc_id = document.get("id")
            video_id = document.get("video_id")
            order = document.get("order", 0)
            embedding = document.get("embedding") or document.get("embedding_azure", [])
            
            if not embedding:
                logger.warning(f"No embedding for document {doc_id}")
                return False
            
            # Normalize embedding for cosine similarity
            embedding_np = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(embedding_np)
            
            # Add to FAISS index
            self.index.add(embedding_np)
            idx = len(self.documents)
            
            # Store document
            self.documents.append(document)
            self._id_to_idx[doc_id] = idx
            
            # Update video-order index
            if video_id:
                if video_id not in self._video_order_index:
                    self._video_order_index[video_id] = {}
                self._video_order_index[video_id][order] = idx
            
            logger.debug(f"Indexed group {doc_id} with order {order}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return False
    
    async def search_by_vector(
        self,
        query_vector: List[float],
        video_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search groups by vector similarity."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query vector
            query_np = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query_np)
            
            # Search
            k = min(limit * 3, self.index.ntotal)  # Get more for filtering
            scores, indices = self.index.search(query_np, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                if score < min_score:
                    continue
                
                doc = self.documents[idx].copy()
                
                # Filter by video_id if specified
                if video_id and doc.get("video_id") != video_id:
                    continue
                
                doc["score"] = float(score)
                results.append(doc)
                
                if len(results) >= limit:
                    break
            
            # Sort by order within video
            results.sort(key=lambda x: (x.get("video_id", ""), x.get("order", 0)))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def get_groups_by_video(
        self,
        video_id: str,
        order_by: str = "order",
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get all groups for a video, ordered by specified field."""
        results = []
        
        for doc in self.documents:
            if doc.get("video_id") == video_id:
                results.append(doc.copy())
        
        # Sort by order_by field
        reverse = not ascending
        results.sort(key=lambda x: x.get(order_by, 0), reverse=reverse)
        
        return results
    
    async def get_group_by_order(
        self,
        video_id: str,
        order: int,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific group by video and order index."""
        if video_id in self._video_order_index:
            idx = self._video_order_index[video_id].get(order)
            if idx is not None and idx < len(self.documents):
                return self.documents[idx].copy()
        
        # Fallback to linear search
        for doc in self.documents:
            if doc.get("video_id") == video_id and doc.get("order") == order:
                return doc.copy()
        
        return None
    
    async def get_adjacent_groups(
        self,
        video_id: str,
        current_order: int,
        direction: str = "both",
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get adjacent groups (previous and/or next)."""
        result: Dict[str, Optional[Dict[str, Any]]] = {
            "previous": None,
            "next": None,
        }
        
        if direction in ("previous", "both"):
            if current_order > 0:
                result["previous"] = await self.get_group_by_order(
                    video_id, current_order - 1
                )
        
        if direction in ("next", "both"):
            result["next"] = await self.get_group_by_order(
                video_id, current_order + 1
            )
        
        return result
    
    async def get_total_groups(
        self,
        video_id: str,
    ) -> int:
        """Get total number of groups for a video."""
        if video_id in self._video_order_index:
            return len(self._video_order_index[video_id])
        
        # Fallback to counting
        return sum(1 for doc in self.documents if doc.get("video_id") == video_id)
    
    async def save(self, path: Optional[str] = None) -> bool:
        """Save index and documents to disk."""
        save_path = Path(path) if path else self.index_path
        if not save_path:
            return False
        
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, str(save_path / "groups.index"))
            
            # Save documents and metadata
            with open(save_path / "groups_docs.json", "w") as f:
                json.dump({
                    "documents": self.documents,
                    "video_order_index": self._video_order_index,
                }, f)
            
            logger.info(f"Saved group index to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    async def load(self, path: Optional[str] = None) -> bool:
        """Load index and documents from disk."""
        load_path = Path(path) if path else self.index_path
        if not load_path:
            return False
        
        try:
            index_file = load_path / "groups.index"
            docs_file = load_path / "groups_docs.json"
            
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
            
            if docs_file.exists():
                with open(docs_file, "r") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self._video_order_index = data.get("video_order_index", {})
                    
                    # Rebuild id index
                    self._id_to_idx = {
                        doc.get("id"): idx 
                        for idx, doc in enumerate(self.documents)
                    }
            
            logger.info(f"Loaded group index from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    async def close(self) -> None:
        """Close provider and save if path is set."""
        if self.index_path:
            await self.save()
        self.index = None
        self.documents = []

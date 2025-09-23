import faiss
import os
import numpy as np
import json
from typing import Optional, Tuple
from loguru import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class FaissIndexManager:
    def __init__(
        self,
        dim: int,
        index_path: str = "faiss_index.index",
        mapping_path: str = "id_mapping.json",
    ):
        self.dim = dim
        self.index_path = os.path.join(BASE_DIR,index_path)
        self.mapping_path = os.path.join(BASE_DIR,mapping_path)
        self.index: Optional[faiss.IndexIDMap] = None
        self.str_to_int_id = {}
        self.int_to_str_id = {}
        self.next_id = 0
        self.load_index_and_mapping()

    def create_index(self):
        try:
            index = faiss.IndexHNSWFlat(self.dim, 32)  # M = 32
            self.index = faiss.IndexIDMap(index)
        except Exception as e:
            raise RuntimeError(f"Failed to create FAISS index: {e}")

    def load_index_and_mapping(self):
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.mapping_path, "r") as f:
                    mapping = json.load(f)
                    self.str_to_int_id = mapping["str_to_int_id"]
                    self.int_to_str_id = {int(k): v for k, v in mapping["int_to_str_id"].items()}
                    self.next_id = max(self.int_to_str_id.keys(), default=-1) + 1

                logger.info("Existing index and mapping loaded")
            else:
                self.create_index()
                logger.info("Created new index and mapping file")
        except Exception as e:
            raise RuntimeError(f"Failed to load index or mapping: {e}")

    def save_index_and_mapping(self):
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.mapping_path, "w") as f:
                json.dump({
                    "str_to_int_id": self.str_to_int_id,
                    "int_to_str_id": self.int_to_str_id
                }, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save index or mapping: {e}")

    def add_embeddings(self, embeddings: np.ndarray, hash_ids: list):
        try:
            int_ids = []
            for hash_id in hash_ids:
                if hash_id not in self.str_to_int_id:
                    int_id = self.next_id
                    self.str_to_int_id[hash_id] = int_id
                    self.int_to_str_id[int_id] = hash_id
                    self.next_id += 1
                else:
                    int_id = self.str_to_int_id[hash_id]
                int_ids.append(int_id)

            int_ids = np.array(int_ids, dtype=np.int64)
            self.index.add_with_ids(embeddings.astype('float32'), int_ids)
            self.save_index_and_mapping()
        except Exception as e:
            raise RuntimeError(f"Failed to add embeddings: {e}")

    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[list, np.ndarray]:
        try:
            distances, int_ids = self.index.search(query_embedding.astype('float32'), k)
            str_ids = [[self.int_to_str_id.get(int_id, None) for int_id in ids] for ids in int_ids]
            return str_ids, distances
        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    def expand_index(self, new_max_elements: int):
        # FAISS HNSWFlat index does not need manual resize
        logger.info("Faiss HNSW index handles dynamic resizing automatically.")


if __name__=='__main__':
    index_manager = FaissIndexManager(dim=1536, index_path="faiss_index.index", mapping_path="id_mapping.json")
    logger.info("Created an instance of the FAISS index manager")

    # No need to explicitly create the index — it's handled internally on init/load
    logger.info("FAISS index created or loaded from disk")

    # Generate random embeddings
    embeddings = np.random.rand(100, 1536).astype(np.float32)
    # logger.info("Generated 100 embeddings of dimension 1536")

    # Generate string hash IDs
    # hash_ids = [f'hash_{i}' for i in range(100)]
    # logger.info("Generated 100 string-based hash IDs")

    # Add embeddings to the index
    # index_manager.add_embeddings(embeddings, hash_ids)
    logger.info("Added embeddings and hash IDs to FAISS index")

    logger.info("QUERY".center(20, '-'))

    # Generate a query embedding
    query = np.array([embeddings[0]])
    # print(query)
    logger.info("Generated a query embedding")

    top_k_hash_ids, distances = index_manager.search(query, k=5)
    logger.info("Performed search and retrieved top K results")

    logger.info(f"Top K Hash IDs: {top_k_hash_ids}")
    logger.info(f"Distances: {distances}")

    # FAISS handles dynamic resizing internally — no need to expand explicitly
    logger.info("FAISS index manages resizing automatically")

    # Save index and mapping
    # index_manager.save_index_and_mapping()
    logger.info("Successfully saved the FAISS index and ID mapping")

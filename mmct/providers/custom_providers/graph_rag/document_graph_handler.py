import os
import networkx as nx
import pickle
from loguru import logger
import numpy as np
from mmct.providers.custom_providers.graph_rag.faiss_index_handler import FaissIndexManager
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DocumentGraph:
    def __init__(self, graph_filename="document_graph.pkl"):
        self.directory = BASE_DIR
        self.graph_filename = graph_filename
        self.graph = self._load_graph()
        self.chapter_embeddings = {
                node_id: self.graph.nodes[node_id]['embeddings']
                for node_id in self.graph.nodes
                if self.graph.nodes[node_id].get('type') == 'chapter'
            }
        
        self.faiss_index_manager = FaissIndexManager(dim=1536)
        logger.info("FASS Indexer loaded")
    def _load_graph(self):
        """Load the graph from a pickle file if it exists, otherwise initialize a new DiGraph."""
        graph_path = os.path.join(self.directory, self.graph_filename)
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                return pickle.load(f)
        else:
            return nx.DiGraph()
        
    def sort_video_documents(self, video_documents: List[dict]) -> List[dict]:
        try:
            if "chapter_transcript" in video_documents[0]:
                sorted(video_documents,key=lambda x: x['chapter_transcript'][:8])
            else:
                raise Exception("Chapter Transcript key is missing from the video documents")

            return video_documents
        except Exception as e:
            logger.exception(f"exception occured while sorting the video documents: {e}")
            raise e

    def save_graph(self):
        """Save the current graph to a pickle file."""
        graph_path = os.path.join(self.directory, self.graph_filename)
        with open(graph_path, "wb") as f:
            pickle.dump(self.graph, f)

    def fetch_chapter_details(self, node_id):
        try:
         
            node_data = self.graph.nodes[node_id]
            chapter_info = {
                "chapter_id": node_id,
                "detailed_summary": node_data.get('detailed_summary'),
                "chapter_transcript": node_data.get('chapter_transcript'),
                "ocr_text_from_scene": node_data.get('ocr_text_from_scene'),
                "youtube_url": node_data.get('youtube_url'),
                "description": node_data.get('description')
            }
            return chapter_info

        except Exception as e:
            raise e
        
    def add_documents(self, video_documents: List[dict], video_id: str):
        """Add a new video document to the graph, updating existing nodes and edges."""
        self.graph.add_node(video_id, type="video")
        video_documents = self.sort_video_documents(video_documents=video_documents)
        node_ids, embeddings_log = list(), list()
        for idx, chapter in enumerate(video_documents):
            chapter_id = f"{video_id}_{idx+1}"
            embeddings = chapter["embeddings"]
            detailed_summary = chapter["detailed_summary"]
            chapter_transcript = chapter["chapter_transcript"]
            ocr_text_from_scene = chapter["text_from_scene"]
            description = chapter['action_taken']
            youtube_url = chapter['youtube_url']
            node_ids.append(chapter_id)
            embeddings_log.append(embeddings)

            # Add chapter node
            self.graph.add_node(
                chapter_id,
                type="chapter",
                embeddings=embeddings,
                detailed_summary=detailed_summary,
                chapter_transcript=chapter_transcript,
                ocr_text_from_scene=ocr_text_from_scene,
                description=description,
                youtube_url=youtube_url
            )

            # Add edge between video and chapter
            self.graph.add_edge(video_id, chapter_id, relation="has_chapter")
            self.graph.add_edge(chapter_id, video_id, relation="has_video_id")

            # Add attribute nodes and edges for chapter
            for key, value in chapter.items():
                if key not in ["embeddings", "detailed_summary", "chapter_transcript", "text_from_scene", "action_taken"]:
                    if value:
                        self.graph.add_node(value, type="attribute")
                        self.graph.add_edge(chapter_id, value, relation=f"has_{key}")
                        self.graph.add_edge(value, chapter_id, relation="has_chapter")

        self.save_graph()
        self.faiss_index_manager.add_embeddings(embeddings=np.array(embeddings_log),hash_ids=node_ids)
        self.chapter_embeddings = {
                node_id: self.graph.nodes[node_id]['embeddings']
                for node_id in self.graph.nodes
                if self.graph.nodes[node_id].get('type') == 'chapter'
            }
        
    def get_chapter_node(self, chapter_id):
        """Retrieve the chapter node and its attributes."""
        if self.graph.has_node(chapter_id):
            return self.graph.nodes[chapter_id]
        else:
            return None

    def fetch_related_chapters(self,node_id, visited_nodes):
        neighbors = []

        # Traverse 1st level neighbors (attributes)
        for attribute_node in self.graph.neighbors(node_id):
            # Traverse 2nd level neighbors (chapters related to the same attribute)
            for related_chapter in self.graph.neighbors(attribute_node):
                if (self.graph.nodes[related_chapter].get("type") == "chapter"
                        and related_chapter != node_id
                        and related_chapter not in visited_nodes):
                    visited_nodes.add(related_chapter)
                    neighbors.append(self.fetch_chapter_details(related_chapter))

        return neighbors
    
    def format_results(self, documents: list) -> list:
        if not documents:
            return []
        result = []
        if isinstance(documents,list):
            for doc in documents:
                res_neighbor = None
                doc['hash_video_id'] = doc['chapter_id'].split('_')[0]
                del doc['chapter_id']
                if 'neighbors' in doc:
                    res_neighbor = self.format_results(doc['neighbors'])
                    del doc['neighbors']
                result.append(doc)
                if res_neighbor:
                    result.extend(res_neighbor)      

        return result
    
    def search(self, query_embedding,top_k,top_n: int):
        try:
            query_embedding = np.array([query_embedding]).reshape(1, -1)
            top_k_hash_ids, distances = self.faiss_index_manager.search(query_embedding=query_embedding,k=top_k)
            top_k_hash_ids = [hid for hid in top_k_hash_ids[0]]
            logger.warning(top_k_hash_ids)
            visited_nodes = set()
            results = []
            
            for node_id in top_k_hash_ids:
                # Add the top match chapter itself
                if node_id in visited_nodes:
                    continue
                visited_nodes.add(node_id)

                chapter_info = self.fetch_chapter_details(node_id=node_id)
                if chapter_info:
                    # Populate its neighbors (related chapters)
                    results.append(chapter_info)
                
                            
            for chapter_info in results:
                neighbors = self.fetch_related_chapters(node_id=chapter_info['chapter_id'], visited_nodes=visited_nodes)
                neighbor_ids = [n['chapter_id'] for n in neighbors if n['chapter_id'] in self.chapter_embeddings]
                neighbor_embeddings = np.array([self.chapter_embeddings[nid] for nid in neighbor_ids])

                if neighbor_embeddings.shape[0] == 0:
                    chapter_info['neighbors'] = []
                    continue

                similarities = cosine_similarity(query_embedding,neighbor_embeddings)[0]
                filtered = sorted(
                    zip(neighbors, similarities),
                    key=lambda x: x[1],
                    reverse=True
                )

                chapter_info['neighbors'] = [n[0] for n in filtered[:top_n]]
                if len(filtered)>top_n:
                    for n in filtered[top_n:]:
                        visited_nodes.remove(n[0]['chapter_id'])
    

            return self.format_results(results)
        except Exception as e:
            logger.exception(f"{e}")
    
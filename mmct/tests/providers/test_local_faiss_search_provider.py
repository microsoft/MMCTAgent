"""
Test suite for LocalFaissSearchProvider.
Tests FAISS indexing, search, and cross-modal retrieval using CLIP embeddings.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from loguru import logger

from mmct.providers.custom_providers.local_faiss_search_provider import LocalFaissSearchProvider


async def main():
    """
    Test function for LocalFaissSearchProvider.
    Uses embeddings from CLIP tests to verify FAISS indexing and search functionality.
    """
    logger.info("Starting LocalFaissSearchProvider test...")
    
    # Check if embedding files exist
    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    required_files = [
        "clip_single_text_embedding.json",
        "clip_batch_text_embeddings.json",
        "clip_single_image_embedding.json",
        "clip_batch_image_embeddings.json",
        "clip_cross_modal_similarity.json"
    ]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(results_dir, f))]
    if missing_files:
        logger.error(f"Missing required embedding files: {missing_files}")
        logger.info("Please run: python mmct/tests/providers/test_image_embedding_provider.py first")
        return
    
    # Initialize provider
    config = {
        "index_path": "../results/test_faiss_indices"
    }
    provider = LocalFaissSearchProvider(config)
    
    # Clean up any existing test indices
    text_index_name = "test-text-index"
    image_index_name = "test-image-index"
    
    logger.info("Cleaning up existing test indices...")
    try:
        if await provider.index_exists(text_index_name):
            await provider.delete_index(text_index_name)
            logger.info(f"Deleted existing index: {text_index_name}")
    except Exception as e:
        logger.debug(f"No existing text index to delete: {e}")
    
    try:
        if await provider.index_exists(image_index_name):
            await provider.delete_index(image_index_name)
            logger.info(f"Deleted existing index: {image_index_name}")
    except Exception as e:
        logger.debug(f"No existing image index to delete: {e}")
    
    test_results = {
        "text_index_tests": {},
        "image_index_tests": {},
        "cross_modal_tests": {}
    }
    
    try:
        # ========================================================================
        # Test 1: Create and populate text embeddings index
        # ========================================================================
        logger.info("\n=== Test 1: Text Embeddings Index ===")
        
        # Load text embeddings
        with open(os.path.join(results_dir, "clip_batch_text_embeddings.json"), 'r') as f:
            text_data = json.load(f)
        
        # Create index
        await provider.create_index(text_index_name, "keyframe")
        logger.info(f"Created text index: {text_index_name}")
        
        # Index text documents
        text_docs = []
        for i, item in enumerate(text_data):
            doc = {
                "id": f"text_{i}",
                "text": item["text"],
                "embeddings": item["embedding"],
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            text_docs.append(doc)
        
        result = await provider.upload_documents(text_docs, text_index_name)
        logger.info(f"Indexed {result['count']} text documents")
        test_results["text_index_tests"]["documents_indexed"] = result['count']
        
        # Verify index exists
        exists = await provider.index_exists(text_index_name)
        logger.info(f"Text index exists: {exists}")
        test_results["text_index_tests"]["index_exists"] = exists
        
        # ========================================================================
        # Test 2: Search text embeddings
        # ========================================================================
        logger.info("\n=== Test 2: Search Text Embeddings ===")
        
        # Load query embedding
        with open(os.path.join(results_dir, "clip_single_text_embedding.json"), 'r') as f:
            query_data = json.load(f)
        
        query_embedding = query_data["embedding"]
        query_text = query_data["text"]
        
        logger.info(f"Query: '{query_text}'")
        
        # Search with embedding
        search_results = await provider.search(
            query=query_text,
            index_name=text_index_name,
            embedding=query_embedding,
            top=3
        )
        
        logger.info(f"Found {len(search_results)} results:")
        test_results["text_index_tests"]["search_results"] = []
        for i, result in enumerate(search_results):
            logger.info(f"  {i+1}. Text: '{result['document']['text']}' (score: {result['score']:.4f})")
            test_results["text_index_tests"]["search_results"].append({
                "rank": i + 1,
                "text": result['document']['text'],
                "score": float(result['score'])
            })
        
        # ========================================================================
        # Test 3: Create and populate image embeddings index
        # ========================================================================
        logger.info("\n=== Test 3: Image Embeddings Index ===")
        
        # Load image embeddings
        with open(os.path.join(results_dir, "clip_batch_image_embeddings.json"), 'r') as f:
            image_data = json.load(f)
        
        # Create index with keyframe schema
        await provider.create_index(image_index_name, "keyframe")
        logger.info(f"Created image index: {image_index_name}")
        
        # Index image documents
        image_docs = []
        for i, item in enumerate(image_data):
            doc = {
                "id": f"image_{i}",
                "video_id": "test_video",
                "keyframe_filename": f"{item['color'].lower()}_frame.jpg",
                "embeddings": item["embedding"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "motion_score": 0.5,
                "timestamp_seconds": float(i),
                "blob_url": "",
                "parent_id": "test_video",
                "parent_duration": 10.0,
                "video_duration": 10.0,
                "color": item["color"]  # Extra field for testing
            }
            image_docs.append(doc)
        
        result = await provider.upload_documents(image_docs, image_index_name)
        logger.info(f"Indexed {result['count']} image documents")
        test_results["image_index_tests"]["documents_indexed"] = result['count']
        
        # Verify index exists
        exists = await provider.index_exists(image_index_name)
        logger.info(f"Image index exists: {exists}")
        test_results["image_index_tests"]["index_exists"] = exists
        
        # ========================================================================
        # Test 4: Search image embeddings
        # ========================================================================
        logger.info("\n=== Test 4: Search Image Embeddings ===")
        
        # Load single image embedding for query
        with open(os.path.join(results_dir, "clip_single_image_embedding.json"), 'r') as f:
            image_query_data = json.load(f)
        
        image_query_embedding = image_query_data["embedding"]
        
        logger.info(f"Query: Single image embedding (dim={len(image_query_embedding)})")
        
        # Search with embedding
        image_search_results = await provider.search(
            query="test image",
            index_name=image_index_name,
            embedding=image_query_embedding,
            top=3
        )
        
        logger.info(f"Found {len(image_search_results)} results:")
        test_results["image_index_tests"]["search_results"] = []
        for i, result in enumerate(image_search_results):
            color = result['document'].get('color', 'Unknown')
            logger.info(f"  {i+1}. Color: {color}, File: {result['document']['keyframe_filename']} (score: {result['score']:.4f})")
            test_results["image_index_tests"]["search_results"].append({
                "rank": i + 1,
                "color": color,
                "filename": result['document']['keyframe_filename'],
                "score": float(result['score'])
            })
        
        # ========================================================================
        # Test 5: Cross-modal search (text query for images)
        # ========================================================================
        logger.info("\n=== Test 5: Cross-Modal Search (Text → Images) ===")
        
        # Load cross-modal data
        with open(os.path.join(results_dir, "clip_cross_modal_similarity.json"), 'r') as f:
            cross_modal_data = json.load(f)
        
        text_query = cross_modal_data["query_text"]
        text_query_embedding = cross_modal_data["query_embedding"]
        
        logger.info(f"Text Query: '{text_query}'")
        logger.info(f"Expected most similar: {cross_modal_data['most_similar_image']}")
        
        # Search images using text embedding
        cross_results = await provider.search(
            query=text_query,
            index_name=image_index_name,
            embedding=text_query_embedding,
            top=3
        )
        
        logger.info(f"Found {len(cross_results)} results:")
        test_results["cross_modal_tests"]["query_text"] = text_query
        test_results["cross_modal_tests"]["expected_most_similar"] = cross_modal_data['most_similar_image']
        test_results["cross_modal_tests"]["search_results"] = []
        
        for i, result in enumerate(cross_results):
            color = result['document'].get('color', 'Unknown')
            logger.info(f"  {i+1}. Color: {color} (score: {result['score']:.4f})")
            test_results["cross_modal_tests"]["search_results"].append({
                "rank": i + 1,
                "color": color,
                "score": float(result['score'])
            })
        
        if cross_results:
            found_color = cross_results[0]['document'].get('color', 'Unknown')
            expected_color = cross_modal_data['most_similar_image']
            test_results["cross_modal_tests"]["top_result"] = found_color
            test_results["cross_modal_tests"]["matches_expected"] = (found_color == expected_color)
            
            if found_color == expected_color:
                logger.info(f"✓ Cross-modal search SUCCESS: Found '{found_color}' as expected!")
            else:
                logger.warning(f"✗ Cross-modal search mismatch: Found '{found_color}', expected '{expected_color}'")
        
        # ========================================================================
        # Test 6: Document existence check
        # ========================================================================
        logger.info("\n=== Test 6: Document Existence Check ===")
        
        exists = await provider.check_is_document_exist("text_0", text_index_name)
        logger.info(f"Document 'text_0' exists: {exists}")
        test_results["text_index_tests"]["doc_exists_check"] = exists
        
        not_exists = await provider.check_is_document_exist("nonexistent_doc", text_index_name)
        logger.info(f"Document 'nonexistent_doc' exists: {not_exists}")
        test_results["text_index_tests"]["doc_not_exists_check"] = not not_exists
        
        # ========================================================================
        # Test 7: Delete document
        # ========================================================================
        logger.info("\n=== Test 7: Delete Document ===")
        
        deleted = await provider.delete_document("text_0", text_index_name)
        logger.info(f"Deleted 'text_0': {deleted}")
        test_results["text_index_tests"]["delete_success"] = deleted
        
        exists_after_delete = await provider.check_is_document_exist("text_0", text_index_name)
        logger.info(f"Document 'text_0' exists after delete: {exists_after_delete}")
        test_results["text_index_tests"]["doc_deleted_verified"] = not exists_after_delete
        
        # ========================================================================
        # Save test results
        # ========================================================================
        output_file = os.path.join(results_dir, "faiss_provider_test_results.json")
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"\n✓ All tests completed!")
        logger.info(f"✓ Results saved to: {output_file}")
        logger.info(f"✓ FAISS index files saved to: {os.path.abspath(config['index_path'])}/")
        
        # Summary
        logger.info("\n=== Test Summary ===")
        logger.info(f"Text documents indexed: {test_results['text_index_tests']['documents_indexed']}")
        logger.info(f"Image documents indexed: {test_results['image_index_tests']['documents_indexed']}")
        logger.info(f"Cross-modal search accuracy: {test_results['cross_modal_tests'].get('matches_expected', False)}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup
        await provider.close()
        logger.info("Provider closed and all indices persisted to disk")


if __name__ == "__main__":
    asyncio.run(main())

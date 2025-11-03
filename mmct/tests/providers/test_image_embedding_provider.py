"""
Test suite for CustomImageEmbeddingProvider.
Tests CLIP embedding generation for both text and images.
"""

import asyncio
import json
import os
import numpy as np
from PIL import Image
from loguru import logger

from mmct.providers.custom_providers.image_embedding_provider import CustomImageEmbeddingProvider


async def main():
    """
    Test function for CLIP embedding provider.
    Creates text and image embeddings and saves them to JSON files.
    """
    logger.info("Starting CLIP embedding provider test...")
    
    # Ensure results directory exists
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Initialize the provider
    config = {
        "model_name": "openai/clip-vit-base-patch32",
        "device": "auto",
        "max_image_size": 224,
        "batch_size": 8
    }

    provider = CustomImageEmbeddingProvider(config)

    results = {
        "model": config["model_name"],
        "device": provider.device,
        "text_embeddings": {},
        "image_embeddings": {},
        "batch_text_embeddings": {},
        "batch_image_embeddings": {}
    }

    try:
        # Test 1: Single text embedding
        logger.info("Testing single text embedding...")
        test_text = "A cat sitting on a couch"
        text_emb = await provider.text_embedding(test_text)
        results["text_embeddings"]["single"] = {
            "text": test_text,
            "embedding_dim": len(text_emb),
            "embedding_sample": text_emb[:5],  # First 5 values for inspection
            "embedding_norm": float(np.linalg.norm(text_emb))
        }
        
        # Save complete single text embedding
        with open(os.path.join(results_dir, "clip_single_text_embedding.json"), 'w') as f:
            json.dump({
                "text": test_text,
                "embedding": text_emb,
                "dimension": len(text_emb),
                "norm": float(np.linalg.norm(text_emb))
            }, f, indent=2)
        
        logger.info(f"Single text embedding: dim={len(text_emb)}, norm={results['text_embeddings']['single']['embedding_norm']:.4f}")
        
        # Test 2: Batch text embeddings
        logger.info("Testing batch text embeddings...")
        test_texts = [
            "A dog playing in the park",
            "A beautiful sunset over the ocean",
            "A person riding a bicycle"
        ]
        batch_text_embs = await provider.batch_text_embedding(test_texts)
        results["batch_text_embeddings"]["count"] = len(batch_text_embs)
        results["batch_text_embeddings"]["samples"] = []
        
        # Save complete batch text embeddings
        batch_text_data = []
        for i, (text, emb) in enumerate(zip(test_texts, batch_text_embs)):
            results["batch_text_embeddings"]["samples"].append({
                "text": text,
                "embedding_dim": len(emb),
                "embedding_sample": emb[:5],
                "embedding_norm": float(np.linalg.norm(emb))
            })
            batch_text_data.append({
                "text": text,
                "embedding": emb,
                "dimension": len(emb),
                "norm": float(np.linalg.norm(emb))
            })
            logger.info(f"Batch text {i+1}: dim={len(emb)}, norm={results['batch_text_embeddings']['samples'][i]['embedding_norm']:.4f}")
        
        with open(os.path.join(results_dir, "clip_batch_text_embeddings.json"), 'w') as f:
            json.dump(batch_text_data, f, indent=2)
        
        # Test 3: Single image embedding (create a test image)
        logger.info("Testing single image embedding...")
        # Create a simple test image
        test_img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        image_emb = await provider.image_embedding(test_img)
        results["image_embeddings"]["single"] = {
            "description": "Solid color test image (224x224)",
            "embedding_dim": len(image_emb),
            "embedding_sample": image_emb[:5],
            "embedding_norm": float(np.linalg.norm(image_emb))
        }
        
        # Save complete single image embedding
        with open(os.path.join(results_dir, "clip_single_image_embedding.json"), 'w') as f:
            json.dump({
                "description": "Solid color test image (224x224, RGB: 73, 109, 137)",
                "embedding": image_emb,
                "dimension": len(image_emb),
                "norm": float(np.linalg.norm(image_emb))
            }, f, indent=2)
        
        logger.info(f"Single image embedding: dim={len(image_emb)}, norm={results['image_embeddings']['single']['embedding_norm']:.4f}")
        
        # Test 4: Batch image embeddings
        logger.info("Testing batch image embeddings...")
        test_images = [
            Image.new('RGB', (224, 224), color=(255, 0, 0)),    # Red
            Image.new('RGB', (224, 224), color=(0, 255, 0)),    # Green
            Image.new('RGB', (224, 224), color=(0, 0, 255))     # Blue
        ]
        batch_image_embs = await provider.batch_image_embedding(test_images)
        results["batch_image_embeddings"]["count"] = len(batch_image_embs)
        results["batch_image_embeddings"]["samples"] = []
        colors = ["Red", "Green", "Blue"]
        
        # Save complete batch image embeddings
        batch_image_data = []
        for i, (color, emb) in enumerate(zip(colors, batch_image_embs)):
            results["batch_image_embeddings"]["samples"].append({
                "description": f"{color} image (224x224)",
                "embedding_dim": len(emb),
                "embedding_sample": emb[:5],
                "embedding_norm": float(np.linalg.norm(emb))
            })
            batch_image_data.append({
                "color": color,
                "description": f"{color} solid color image (224x224)",
                "embedding": emb,
                "dimension": len(emb),
                "norm": float(np.linalg.norm(emb))
            })
            logger.info(f"Batch image {i+1} ({color}): dim={len(emb)}, norm={results['batch_image_embeddings']['samples'][i]['embedding_norm']:.4f}")
        
        with open(os.path.join(results_dir, "clip_batch_image_embeddings.json"), 'w') as f:
            json.dump(batch_image_data, f, indent=2)
        
        # Test 5: Text-to-Image similarity
        logger.info("Testing text-to-image similarity...")
        query_text = "A red image"
        query_emb = await provider.text_embedding(query_text)
        
        similarities = []
        for i, (color, img_emb) in enumerate(zip(colors, batch_image_embs)):
            # Compute cosine similarity
            similarity = float(np.dot(query_emb, img_emb))
            similarities.append({
                "image": color,
                "similarity": similarity
            })
            logger.info(f"Similarity between '{query_text}' and {color} image: {similarity:.4f}")
        
        results["cross_modal_similarity"] = {
            "query_text": query_text,
            "similarities": similarities,
            "most_similar": max(similarities, key=lambda x: x["similarity"])
        }
        
        # Save complete cross-modal similarity data
        with open(os.path.join(results_dir, "clip_cross_modal_similarity.json"), 'w') as f:
            json.dump({
                "query_text": query_text,
                "query_embedding": query_emb,
                "query_dimension": len(query_emb),
                "query_norm": float(np.linalg.norm(query_emb)),
                "image_comparisons": [
                    {
                        "image_color": color,
                        "image_embedding": img_emb,
                        "cosine_similarity": float(np.dot(query_emb, img_emb))
                    }
                    for color, img_emb in zip(colors, batch_image_embs)
                ],
                "most_similar_image": max(similarities, key=lambda x: x["similarity"])["image"]
            }, f, indent=2)
        
        # Save results to JSON file
        output_file = os.path.join(results_dir, "clip_embeddings_test_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✓ All tests completed successfully!")
        logger.info(f"✓ Summary saved to: {output_file}")
        logger.info(f"✓ Complete embeddings saved to:")
        logger.info(f"  - {os.path.join(results_dir, 'clip_single_text_embedding.json')}")
        logger.info(f"  - {os.path.join(results_dir, 'clip_batch_text_embeddings.json')}")
        logger.info(f"  - {os.path.join(results_dir, 'clip_single_image_embedding.json')}")
        logger.info(f"  - {os.path.join(results_dir, 'clip_batch_image_embeddings.json')}")
        logger.info(f"  - {os.path.join(results_dir, 'clip_cross_modal_similarity.json')}")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        provider.close()
        logger.info("Provider closed and resources cleaned up")


if __name__ == "__main__":
    asyncio.run(main())

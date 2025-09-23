#!/usr/bin/env python3
"""
Quick Vector Database Test
A lightweight test to verify the setup is working correctly.
"""

import time
import random
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import argparse


def quick_test(host="localhost", port=6333, collection_name="quick_test"):
    """Run a quick test to verify the setup"""
    print("Quick Vector Database Test")
    print("=" * 30)
    
    try:
        # Connect to Qdrant
        print("Connecting to Qdrant...")
        client = QdrantClient(host=host, port=port)
        
        # Test connection
        health = client.get_collections()
        print("✓ Connected to Qdrant successfully")
        
        # Create test collection
        print("Creating test collection...")
        try:
            client.delete_collection(collection_name)
        except:
            pass  # Collection might not exist
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE
            )
        )
        print("✓ Test collection created")
        
        # Generate test data
        print("Generating test data...")
        test_points = []
        for i in range(1000):
            vector = np.random.normal(0, 1, 768)
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            point = PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "id": i,
                    "text": f"Test document {i}",
                    "category": random.choice(["A", "B", "C"])
                }
            )
            test_points.append(point)
        
        # Insert test data
        print("Inserting test data...")
        start_time = time.time()
        client.upsert(
            collection_name=collection_name,
            points=test_points
        )
        insert_time = time.time() - start_time
        print(f"✓ Inserted 1000 points in {insert_time:.3f}s ({1000/insert_time:.0f} points/sec)")
        
        # Test search
        print("Testing search...")
        query_vector = np.random.normal(0, 1, 768)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        start_time = time.time()
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=10
        )
        search_time = time.time() - start_time
        print(f"✓ Search completed in {search_time:.3f}s")
        print(f"  Found {len(results)} results")
        
        # Test batch search
        print("Testing batch search...")
        query_vectors = [np.random.normal(0, 1, 768) for _ in range(5)]
        query_vectors = [v / np.linalg.norm(v) for v in query_vectors]
        
        start_time = time.time()
        batch_results = client.search_batch(
            collection_name=collection_name,
            requests=[{"vector": v.tolist(), "limit": 5} for v in query_vectors]
        )
        batch_time = time.time() - start_time
        print(f"✓ Batch search completed in {batch_time:.3f}s")
        print(f"  Processed {len(batch_results)} queries")
        
        # Test retrieve by ID
        print("Testing retrieve by ID...")
        test_ids = [0, 100, 500, 999]
        start_time = time.time()
        retrieved = client.retrieve(
            collection_name=collection_name,
            ids=test_ids
        )
        retrieve_time = time.time() - start_time
        print(f"✓ Retrieved {len(retrieved)} points by ID in {retrieve_time:.3f}s")
        
        # Get collection info
        print("Getting collection info...")
        collection_info = client.get_collection(collection_name)
        print(f"✓ Collection has {collection_info.points_count} points")
        
        # Cleanup
        print("Cleaning up...")
        client.delete_collection(collection_name)
        print("✓ Test collection deleted")
        
        print("\n" + "=" * 30)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("=" * 30)
        print(f"Insert Performance: {1000/insert_time:.0f} points/sec")
        print(f"Search Performance: {1/search_time:.0f} queries/sec")
        print(f"Batch Search: {len(query_vectors)/batch_time:.0f} queries/sec")
        print(f"Retrieve by ID: {len(test_ids)/retrieve_time:.0f} operations/sec")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Quick vector database test")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="quick_test", help="Test collection name")
    
    args = parser.parse_args()
    
    success = quick_test(
        host=args.host,
        port=args.port,
        collection_name=args.collection
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

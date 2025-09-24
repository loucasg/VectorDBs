#!/usr/bin/env python3
"""
Vector Database Population Script
Populates Qdrant with 10 million chunk records for testing purposes.
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
import argparse


class VectorDatabasePopulator:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "test_vectors"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_dim = 1024  # Common embedding dimension (e.g., BERT, sentence-transformers)
        self.batch_size = 1000  # Batch size for efficient insertion
        
    def create_collection(self):
        """Create the collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            if any(col.name == self.collection_name for col in collections.collections):
                print(f"Collection '{self.collection_name}' already exists. Adding to existing collection...")
                return
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection '{self.collection_name}' with {self.vector_dim}D vectors")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def generate_random_vector(self) -> List[float]:
        """Generate a random normalized vector"""
        vector = np.random.normal(0, 1, self.vector_dim)
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def generate_chunk_data(self, chunk_id: int) -> Dict[str, Any]:
        """Generate sample chunk data"""
        return {
            "id": chunk_id,
            "text": f"This is chunk number {chunk_id} with some sample content. " * random.randint(1, 5),
            "metadata": {
                "source": f"document_{chunk_id % 1000}",
                "page": random.randint(1, 100),
                "section": random.choice(["introduction", "body", "conclusion"]),
                "timestamp": int(time.time()) + chunk_id,
                "category": random.choice(["technical", "business", "legal", "scientific"]),
                "confidence": round(random.uniform(0.5, 1.0), 3)
            }
        }

    def get_next_available_id(self) -> int:
        """Get the next available ID to avoid conflicts when adding to existing data"""
        try:
            # Get collection info to find the highest existing ID
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                return 0
            
            # Use timestamp-based ID generation to avoid conflicts
            # This ensures unique IDs even across multiple runs
            import time
            timestamp_id = int(time.time() * 1000000)  # Microsecond timestamp
            return timestamp_id
        except Exception as e:
            print(f"Error getting next available ID: {e}")
            return 0  # Fallback to starting from 0

    def create_batch_points(self, start_id: int, batch_size: int) -> List[PointStruct]:
        """Create a batch of points for insertion"""
        points = []
        for i in range(batch_size):
            chunk_id = start_id + i
            chunk_data = self.generate_chunk_data(chunk_id)
            vector = self.generate_random_vector()
            
            point = PointStruct(
                id=chunk_id,
                vector=vector,
                payload=chunk_data
            )
            points.append(point)
        
        return points

    def insert_batch(self, points: List[PointStruct]) -> bool:
        """Insert a batch of points"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"Error inserting batch: {e}")
            return False

    def populate_database(self, total_records: int = 10_000_000, max_workers: int = 4):
        """Populate the database with the specified number of records"""
        print(f"Starting population of {total_records:,} records...")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max workers: {max_workers}")
        
        # Get the next available ID to avoid conflicts
        start_id = self.get_next_available_id()
        print(f"Starting from ID: {start_id}")
        
        start_time = time.time()
        total_batches = (total_records + self.batch_size - 1) // self.batch_size
        
        # Create progress bar
        pbar = tqdm(total=total_records, desc="Inserting records", unit="records")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        successful_batches = 0
        failed_batches = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for batch_idx in range(total_batches):
                batch_start_id = start_id + (batch_idx * self.batch_size)
                current_batch_size = min(self.batch_size, total_records - (batch_idx * self.batch_size))
                
                # Create batch points
                points = self.create_batch_points(batch_start_id, current_batch_size)
                
                # Submit batch for insertion
                future = executor.submit(self.insert_batch, points)
                futures.append(future)
                
                # Process completed futures
                if len(futures) >= max_workers * 2:  # Keep some futures in flight
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            if future.result():
                                successful_batches += 1
                            else:
                                failed_batches += 1
                            completed_futures.append(future)
                    
                    # Remove completed futures
                    for future in completed_futures:
                        futures.remove(future)
                        pbar.update(self.batch_size)
            
            # Wait for remaining futures
            for future in futures:
                if future.result():
                    successful_batches += 1
                else:
                    failed_batches += 1
                pbar.update(self.batch_size)
        
        pbar.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Print statistics
        print("\n" + "="*50)
        print("POPULATION COMPLETED")
        print("="*50)
        print(f"Total records: {total_records:,}")
        print(f"Successful batches: {successful_batches:,}")
        print(f"Failed batches: {failed_batches:,}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Records per second: {total_records / duration:,.0f}")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory: {final_memory:.2f} MB")
        
        # Verify collection info
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"Collection points count: {collection_info.points_count:,}")
        except Exception as e:
            print(f"Error getting collection info: {e}")

    def get_collection_stats(self):
        """Get and display collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print("\n" + "="*50)
            print("COLLECTION STATISTICS")
            print("="*50)
            print(f"Collection name: {self.collection_name}")
            print(f"Points count: {collection_info.points_count:,}")
            print(f"Vector size: {collection_info.config.params.vectors.size}")
            print(f"Distance metric: {collection_info.config.params.vectors.distance}")
            print(f"Indexed vectors: {collection_info.indexed_vectors_count:,}")
        except Exception as e:
            print(f"Error getting collection stats: {e}")


def main():
    parser = argparse.ArgumentParser(description="Populate vector database with test data")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="test_vectors", help="Collection name")
    parser.add_argument("--records", type=int, default=10_000_000, help="Number of records to insert")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    
    args = parser.parse_args()
    
    # Create populator instance
    populator = VectorDatabasePopulator(
        host=args.host,
        port=args.port,
        collection_name=args.collection
    )
    populator.vector_dim = args.vector_dim
    
    try:
        # Create collection
        populator.create_collection()
        
        # Populate database
        populator.populate_database(
            total_records=args.records,
            max_workers=args.workers
        )
        
        # Show final stats
        populator.get_collection_stats()
        
    except KeyboardInterrupt:
        print("\nPopulation interrupted by user")
    except Exception as e:
        print(f"Error during population: {e}")
        raise


if __name__ == "__main__":
    main()

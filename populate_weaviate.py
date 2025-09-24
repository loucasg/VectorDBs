#!/usr/bin/env python3
"""
Weaviate Vector Database Population Script

This script populates a Weaviate collection with test vector data.
It supports incremental population (adds to existing collections) and
different vector dimensions.
"""

import argparse
import time
import numpy as np
import psutil
from tqdm import tqdm
import weaviate
import weaviate.classes as wvc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class WeaviatePopulator:
    def __init__(self, host="localhost", port="8080", collection_name="TestVectors", 
                 vector_dim=1024, batch_size=1000, max_workers=4):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.client = None
        
    def connect(self):
        """Connect to Weaviate server"""
        try:
            self.client = weaviate.Client(
                url=f"http://{self.host}:{self.port}",
                additional_headers={"X-OpenAI-Api-Key": "dummy"}  # Required even if not using OpenAI
            )
            
            # Test connection
            if self.client.is_ready():
                print(f"✅ Connected to Weaviate at {self.host}:{self.port}")
                return True
            else:
                print(f"❌ Weaviate server not ready")
                return False
        except Exception as e:
            print(f"❌ Error connecting to Weaviate: {e}")
            return False
    
    def create_collection(self):
        """Create collection if it doesn't exist, or check if it exists"""
        try:
            # Check if collection exists
            if self.client.schema.exists(self.collection_name):
                print(f"Collection '{self.collection_name}' already exists. Adding to existing collection...")
                
                # Get the vector dimension from existing collection
                schema = self.client.schema.get(self.collection_name)
                vectorizer_config = schema.get("vectorizer", {})
                if "vectorIndexConfig" in vectorizer_config:
                    existing_dim = vectorizer_config["vectorIndexConfig"].get("vectorCacheMaxObjects", 0)
                    if existing_dim and existing_dim != self.vector_dim:
                        print(f"⚠️  Warning: Collection may have different vector dimension than {self.vector_dim}")
                
                return True
            else:
                # Create new collection
                print(f"Creating new collection '{self.collection_name}'...")
                
                # Define collection schema
                collection_schema = {
                    "class": self.collection_name,
                    "description": f"Collection for {self.collection_name}",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "vectorIndexType": "hnsw",
                    "vectorIndexConfig": {
                        "distance": "cosine",
                        "ef": 64,
                        "efConstruction": 128,
                        "maxConnections": 16,
                        "vectorCacheMaxObjects": 1000000
                    },
                    "properties": [
                        {
                            "name": "text_content",
                            "dataType": ["text"],
                            "description": "Text content of the document"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"],
                            "description": "Metadata as JSON string"
                        }
                    ]
                }
                
                self.client.schema.create_class(collection_schema)
                print(f"✅ Collection '{self.collection_name}' created successfully")
                return True
                
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return False
    
    def get_collection_count(self):
        """Get the current count of objects in the collection"""
        try:
            result = self.client.query.aggregate(self.collection_name).with_meta_count().do()
            if "data" in result and "Aggregate" in result["data"]:
                count = result["data"]["Aggregate"][self.collection_name][0]["meta"]["count"]
                return count
            return 0
        except Exception as e:
            print(f"Error getting collection count: {e}")
            return 0
    
    def generate_random_vector(self):
        """Generate a random normalized vector"""
        vector = np.random.random(self.vector_dim).astype(np.float32)
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    def generate_batch_data(self, batch_size):
        """Generate a batch of data for insertion"""
        objects = []
        
        for i in range(batch_size):
            vector = self.generate_random_vector()
            text_content = f"Test document {int(time.time() * 1000000) + i}"
            metadata = json.dumps({
                "source": "test",
                "batch": int(time.time()),
                "index": i,
                "vector_dim": self.vector_dim
            })
            
            objects.append({
                "vector": vector,
                "text_content": text_content,
                "metadata": metadata
            })
        
        return objects
    
    def insert_batch(self, objects):
        """Insert a batch of data into Weaviate"""
        try:
            # Use batch insert
            with self.client.batch as batch:
                for obj in objects:
                    batch.add_data_object(
                        data_object={
                            "text_content": obj["text_content"],
                            "metadata": obj["metadata"]
                        },
                        class_name=self.collection_name,
                        vector=obj["vector"]
                    )
            
            return True, len(objects)
        except Exception as e:
            print(f"❌ Error inserting batch: {e}")
            return False, 0
    
    def populate_database(self, num_records):
        """Populate the database with the specified number of records"""
        print(f"\n{'='*50}")
        print(f"POPULATING WEAVIATE DATABASE")
        print(f"{'='*50}")
        print(f"Collection: {self.collection_name}")
        print(f"Records to insert: {num_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max workers: {self.max_workers}")
        
        # Get current count
        current_count = self.get_collection_count()
        print(f"Current collection count: {current_count:,}")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        successful_batches = 0
        failed_batches = 0
        total_inserted = 0
        
        # Calculate number of batches
        num_batches = (num_records + self.batch_size - 1) // self.batch_size
        
        print(f"\nStarting population of {num_records:,} records...")
        print(f"Processing {num_batches} batches...")
        
        with tqdm(total=num_records, desc="Inserting records") as pbar:
            for batch_idx in range(num_batches):
                # Calculate batch size for this iteration
                current_batch_size = min(self.batch_size, num_records - total_inserted)
                
                # Generate batch data
                objects = self.generate_batch_data(current_batch_size)
                
                # Insert batch
                success, inserted_count = self.insert_batch(objects)
                
                if success:
                    successful_batches += 1
                    total_inserted += inserted_count
                else:
                    failed_batches += 1
                
                pbar.update(inserted_count)
                
                # Break if we've inserted enough records
                if total_inserted >= num_records:
                    break
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Get final collection count
        final_count = self.get_collection_count()
        
        print(f"\n{'='*50}")
        print(f"POPULATION COMPLETED")
        print(f"{'='*50}")
        print(f"Total records inserted: {total_inserted:,}")
        print(f"Successful batches: {successful_batches}")
        print(f"Failed batches: {failed_batches}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Records per second: {total_inserted / duration:.0f}")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        
        print(f"\n{'='*50}")
        print(f"COLLECTION STATISTICS")
        print(f"{'='*50}")
        print(f"Collection name: {self.collection_name}")
        print(f"Total objects: {final_count:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Index type: HNSW")
        print(f"Distance metric: COSINE")


def main():
    parser = argparse.ArgumentParser(description="Populate Weaviate with test vector data")
    parser.add_argument("--host", default="localhost", help="Weaviate host (default: localhost)")
    parser.add_argument("--port", default="8080", help="Weaviate port (default: 8080)")
    parser.add_argument("--collection", default="TestVectors", help="Collection name (default: TestVectors)")
    parser.add_argument("--records", type=int, default=10000, help="Number of records to insert (default: 10000)")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts (default: 1000)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    
    args = parser.parse_args()
    
    # Create populator instance
    populator = WeaviatePopulator(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        vector_dim=args.vector_dim,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    
    # Connect to Weaviate
    if not populator.connect():
        return 1
    
    try:
        # Create collection
        if not populator.create_collection():
            return 1
        
        # Populate database
        populator.populate_database(args.records)
        
        print(f"\n✅ Successfully populated Weaviate collection '{args.collection}' with {args.records:,} records")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Population interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during population: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

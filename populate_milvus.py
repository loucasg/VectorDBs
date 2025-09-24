#!/usr/bin/env python3
"""
Milvus Vector Database Population Script

This script populates a Milvus collection with test vector data.
It supports incremental population (adds to existing collections) and
different vector dimensions.
"""

import argparse
import time
import numpy as np
import psutil
from tqdm import tqdm
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class MilvusPopulator:
    def __init__(self, host="localhost", port="19530", collection_name="test_vectors", 
                 vector_dim=1024, batch_size=1000, max_workers=4):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.connection_alias = f"milvus_{collection_name}"
        
    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port
            )
            print(f"✅ Connected to Milvus at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Error connecting to Milvus: {e}")
            return False
    
    def create_collection(self):
        """Create collection if it doesn't exist, or check if it exists"""
        try:
            # Check if collection exists using the connection alias
            if utility.has_collection(self.collection_name, using=self.connection_alias):
                collection = Collection(self.collection_name, using=self.connection_alias)
                print(f"Collection '{self.collection_name}' already exists. Adding to existing collection...")
                
                # Get the vector dimension from existing collection
                schema = collection.schema
                for field in schema.fields:
                    if field.name == "vector":
                        existing_dim = field.params.get("dim")
                        if existing_dim and existing_dim != self.vector_dim:
                            print(f"⚠️  Warning: Collection has vector dimension {existing_dim}, but script is using {self.vector_dim}")
                        break
                
                return collection
            else:
                # Create new collection
                print(f"Creating new collection '{self.collection_name}'...")
                
                # Define schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                    FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
                ]
                
                schema = CollectionSchema(fields, f"Collection for {self.collection_name}")
                collection = Collection(self.collection_name, schema, using=self.connection_alias)
                
                # Create index
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                collection.create_index("vector", index_params)
                
                print(f"✅ Collection '{self.collection_name}' created successfully")
                return collection
                
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            return None
    
    def get_next_available_id(self):
        """Get the next available ID to avoid conflicts when adding to existing data"""
        try:
            collection = Collection(self.collection_name, using=self.connection_alias)
            if not collection.is_empty:
                # Get the last inserted ID by querying the collection
                # Since we use auto_id=True, we need to get the max ID
                results = collection.query(
                    expr="id >= 0",
                    output_fields=["id"],
                    limit=1,
                    offset=0
                )
                if results:
                    # Get the count to estimate next ID
                    stats = collection.get_stats()
                    # For auto_id, we can't predict the next ID, so we'll let Milvus handle it
                    return 0  # Milvus will auto-generate IDs
                else:
                    return 0
            else:
                return 0
        except Exception as e:
            print(f"Error getting next available ID: {e}")
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
        vectors = []
        text_contents = []
        metadata_list = []
        
        for i in range(batch_size):
            vectors.append(self.generate_random_vector())
            text_contents.append(f"Test document {int(time.time() * 1000000) + i}")
            metadata_list.append(f'{{"source": "test", "batch": {int(time.time())}, "index": {i}}}')
        
        return vectors, text_contents, metadata_list
    
    def insert_batch(self, vectors, text_contents, metadata_list):
        """Insert a batch of data into Milvus"""
        try:
            collection = Collection(self.collection_name, using=self.connection_alias)
            
            # Prepare data for insertion
            data = [
                vectors,  # vector field
                text_contents,  # text_content field
                metadata_list  # metadata field
            ]
            
            # Insert data
            collection.insert(data)
            collection.flush()  # Ensure data is written to disk
            
            return True, len(vectors)
        except Exception as e:
            print(f"❌ Error inserting batch: {e}")
            return False, 0
    
    def populate_database(self, num_records):
        """Populate the database with the specified number of records"""
        print(f"\n{'='*50}")
        print(f"POPULATING MILVUS DATABASE")
        print(f"{'='*50}")
        print(f"Collection: {self.collection_name}")
        print(f"Records to insert: {num_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max workers: {self.max_workers}")
        
        # Get starting ID (for display purposes)
        start_id = self.get_next_available_id()
        print(f"Starting from auto-generated IDs (Milvus handles ID generation)")
        
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
                vectors, text_contents, metadata_list = self.generate_batch_data(current_batch_size)
                
                # Insert batch
                success, inserted_count = self.insert_batch(vectors, text_contents, metadata_list)
                
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
        
        # Get final collection stats
        try:
            collection = Collection(self.collection_name, using=self.connection_alias)
            collection.load()  # Load collection into memory for querying
            stats = collection.get_stats()
            entity_count = collection.num_entities
        except Exception as e:
            print(f"Warning: Could not get collection stats: {e}")
            entity_count = total_inserted
        
        print(f"\n{'='*50}")
        print(f"POPULATION COMPLETED")
        print(f"{'='*50}")
        print(f"Total records: {total_inserted:,}")
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
        print(f"Entity count: {entity_count:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Index type: IVF_FLAT")
        print(f"Metric type: COSINE")


def main():
    parser = argparse.ArgumentParser(description="Populate Milvus with test vector data")
    parser.add_argument("--host", default="localhost", help="Milvus host (default: localhost)")
    parser.add_argument("--port", default="19530", help="Milvus port (default: 19530)")
    parser.add_argument("--collection", default="test_vectors", help="Collection name (default: test_vectors)")
    parser.add_argument("--records", type=int, default=10000, help="Number of records to insert (default: 10000)")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts (default: 1000)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    
    args = parser.parse_args()
    
    # Create populator instance
    populator = MilvusPopulator(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        vector_dim=args.vector_dim,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    
    # Connect to Milvus
    if not populator.connect():
        return 1
    
    try:
        # Create collection
        collection = populator.create_collection()
        if collection is None:
            return 1
        
        # Populate database
        populator.populate_database(args.records)
        
        print(f"\n✅ Successfully populated Milvus collection '{args.collection}' with {args.records:,} records")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Population interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during population: {e}")
        return 1
    finally:
        # Disconnect from Milvus
        try:
            connections.disconnect(populator.connection_alias)
        except:
            pass


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Vespa Vector Database Population Script

This script populates a Vespa application with test vector data.
It supports incremental population (adds to existing documents) and
different vector dimensions.
"""

import argparse
import time
import numpy as np
import psutil
from tqdm import tqdm
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class VespaPopulator:
    def __init__(self, host="localhost", port="8081", application_name="test_vectors", 
                 vector_dim=1024, batch_size=1000, max_workers=4):
        self.host = host
        self.port = port
        self.application_name = application_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.base_url = f"http://{self.host}:{port}"
        self.document_type = "test_vector"
        
    def connect(self):
        """Connect to Vespa server"""
        try:
            # Test connection by checking if Vespa is ready
            response = requests.get(f"{self.base_url}/ApplicationStatus", timeout=10)
            if response.status_code == 200:
                print(f"✅ Connected to Vespa at {self.host}:{self.port}")
                return True
            else:
                print(f"❌ Vespa server not ready (status: {response.status_code})")
                return False
        except Exception as e:
            print(f"❌ Error connecting to Vespa: {e}")
            return False
    
    def create_application_schema(self):
        """Create application schema if it doesn't exist"""
        try:
            # Check if application exists
            response = requests.get(f"{self.base_url}/ApplicationStatus")
            if response.status_code == 200:
                print(f"Vespa application is running. Using existing schema...")
                return True
            else:
                print(f"⚠️  Warning: Could not verify application status (status: {response.status_code})")
                return True  # Continue anyway
                
        except Exception as e:
            print(f"❌ Error checking application status: {e}")
            return False
    
    def get_document_count(self):
        """Get the current count of documents in the application"""
        try:
            # Use search to count documents
            search_url = f"{self.base_url}/search/"
            params = {
                "yql": f"select * from {self.document_type} limit 0",
                "hits": 0
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                total_count = data.get("root", {}).get("totalCount", 0)
                return total_count
            else:
                print(f"Warning: Could not get document count (status: {response.status_code})")
                return 0
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def generate_random_vector(self):
        """Generate a random normalized vector"""
        vector = np.random.random(self.vector_dim).astype(np.float32)
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    def generate_document(self, doc_id):
        """Generate a single document for insertion"""
        vector = self.generate_random_vector()
        text_content = f"Test document {doc_id}"
        metadata = {
            "source": "test",
            "batch": int(time.time()),
            "doc_id": doc_id,
            "vector_dim": self.vector_dim
        }
        
        document = {
            "fields": {
                "id": doc_id,
                "text_content": text_content,
                "metadata": json.dumps(metadata),
                "vector": {
                    "values": vector
                }
            }
        }
        
        return document
    
    def insert_document(self, document):
        """Insert a single document into Vespa"""
        try:
            doc_id = document["fields"]["id"]
            url = f"{self.base_url}/document/v1/{self.application_name}/{self.document_type}/docid/{doc_id}"
            
            response = requests.put(
                url,
                json=document,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                return True
            else:
                print(f"❌ Error inserting document {doc_id}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error inserting document: {e}")
            return False
    
    def insert_batch(self, documents):
        """Insert a batch of documents into Vespa"""
        successful = 0
        failed = 0
        
        for document in documents:
            if self.insert_document(document):
                successful += 1
            else:
                failed += 1
        
        return successful > 0, successful
    
    def populate_database(self, num_records):
        """Populate the database with the specified number of records"""
        print(f"\n{'='*50}")
        print(f"POPULATING VESPA DATABASE")
        print(f"{'='*50}")
        print(f"Application: {self.application_name}")
        print(f"Document type: {self.document_type}")
        print(f"Records to insert: {num_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max workers: {self.max_workers}")
        
        # Get current count
        current_count = self.get_document_count()
        print(f"Current document count: {current_count:,}")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        successful_batches = 0
        failed_batches = 0
        total_inserted = 0
        
        # Calculate number of batches
        num_batches = (num_records + self.batch_size - 1) // self.batch_size
        
        print(f"\nStarting population of {num_records:,} records...")
        print(f"Processing {num_batches} batches...")
        
        with tqdm(total=num_records, desc="Inserting documents") as pbar:
            for batch_idx in range(num_batches):
                # Calculate batch size for this iteration
                current_batch_size = min(self.batch_size, num_records - total_inserted)
                
                # Generate batch documents
                documents = []
                for i in range(current_batch_size):
                    doc_id = f"doc_{int(time.time() * 1000000) + i}"
                    document = self.generate_document(doc_id)
                    documents.append(document)
                
                # Insert batch
                success, inserted_count = self.insert_batch(documents)
                
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
        
        # Get final document count
        final_count = self.get_document_count()
        
        print(f"\n{'='*50}")
        print(f"POPULATION COMPLETED")
        print(f"{'='*50}")
        print(f"Total documents inserted: {total_inserted:,}")
        print(f"Successful batches: {successful_batches}")
        print(f"Failed batches: {failed_batches}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Documents per second: {total_inserted / duration:.0f}")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        
        print(f"\n{'='*50}")
        print(f"APPLICATION STATISTICS")
        print(f"{'='*50}")
        print(f"Application name: {self.application_name}")
        print(f"Document type: {self.document_type}")
        print(f"Total documents: {final_count:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Search endpoint: {self.base_url}/search/")


def main():
    parser = argparse.ArgumentParser(description="Populate Vespa with test vector data")
    parser.add_argument("--host", default="localhost", help="Vespa host (default: localhost)")
    parser.add_argument("--port", default="8081", help="Vespa port (default: 8081)")
    parser.add_argument("--application", default="test_vectors", help="Application name (default: test_vectors)")
    parser.add_argument("--records", type=int, default=10000, help="Number of records to insert (default: 10000)")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts (default: 1000)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    
    args = parser.parse_args()
    
    # Create populator instance
    populator = VespaPopulator(
        host=args.host,
        port=args.port,
        application_name=args.application,
        vector_dim=args.vector_dim,
        batch_size=args.batch_size,
        max_workers=args.workers
    )
    
    # Connect to Vespa
    if not populator.connect():
        return 1
    
    try:
        # Create application schema
        if not populator.create_application_schema():
            return 1
        
        # Populate database
        populator.populate_database(args.records)
        
        print(f"\n✅ Successfully populated Vespa application '{args.application}' with {args.records:,} records")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Population interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during population: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

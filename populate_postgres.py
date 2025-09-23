#!/usr/bin/env python3
"""
PostgreSQL Vector Database Population Script
Populates PostgreSQL with pgvector extension with test data.
"""

import asyncio
import time
import random
import numpy as np
import json
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
import argparse
from tqdm import tqdm


class PostgreSQLVectorPopulator:
    def __init__(self, host: str = "localhost", port: int = 5432, 
                 database: str = "vectordb", user: str = "postgres", 
                 password: str = "postgres"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.vector_dim = 768
        self.batch_size = 1000
        
    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
    
    def create_collection(self):
        """Create the vector embeddings table if it doesn't exist"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'vector_embeddings'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    print("Table 'vector_embeddings' already exists. Adding to existing data...")
                else:
                    print("Table 'vector_embeddings' does not exist. Creating it...")
                    # The table should already be created by init-postgres.sql
                    # But let's create it just in case
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS vector_embeddings (
                            id SERIAL PRIMARY KEY,
                            vector_id INTEGER UNIQUE NOT NULL,
                            embedding VECTOR(768),
                            text_content TEXT,
                            metadata JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                
                conn.commit()
                print(f"Table ready for {self.vector_dim}D vectors")
                
        except Exception as e:
            print(f"Error setting up table: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

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
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) + 1 FROM vector_embeddings;")
                next_id = cur.fetchone()[0]
                return next_id
        except Exception as e:
            print(f"Error getting next available ID: {e}")
            return 1  # Fallback to starting from 1
        finally:
            conn.close()

    def insert_batch(self, batch_data: List[Dict[str, Any]]) -> bool:
        """Insert a batch of records"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare batch insert
                insert_query = """
                    INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (vector_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        text_content = EXCLUDED.text_content,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """
                
                # Convert data to tuples for batch insert
                batch_tuples = []
                for record in batch_data:
                    vector = self.generate_random_vector()
                    chunk_data = self.generate_chunk_data(record['id'])
                    
                    batch_tuples.append((
                        record['id'],
                        vector,
                        chunk_data['text'],
                        json.dumps(chunk_data['metadata'])  # Convert dict to JSON string
                    ))
                
                # Execute batch insert
                cur.executemany(insert_query, batch_tuples)
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error inserting batch: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

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
                
                # Create batch data
                batch_data = [{"id": batch_start_id + i} for i in range(current_batch_size)]
                
                # Submit batch for insertion
                future = executor.submit(self.insert_batch, batch_data)
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
        self.get_collection_stats()

    def get_collection_stats(self):
        """Get and display collection statistics"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get basic stats
                cur.execute("SELECT COUNT(*) as total_points FROM vector_embeddings;")
                total_points = cur.fetchone()['total_points']
                
                # Get vector dimension (pgvector stores dimensions in the type definition)
                vector_dim = self.vector_dim  # We know it's 768 from our setup
                
                # Get metadata stats
                cur.execute("""
                    SELECT 
                        AVG(LENGTH(metadata::TEXT)) as avg_metadata_size,
                        MIN(created_at) as earliest_record,
                        MAX(created_at) as latest_record
                    FROM vector_embeddings;
                """)
                metadata_stats = cur.fetchone()
                
                print("\n" + "="*50)
                print("COLLECTION STATISTICS")
                print("="*50)
                print(f"Total points: {total_points:,}")
                print(f"Vector dimensions: {vector_dim}")
                print(f"Average metadata size: {metadata_stats['avg_metadata_size']:.1f} characters")
                print(f"Earliest record: {metadata_stats['earliest_record']}")
                print(f"Latest record: {metadata_stats['latest_record']}")
                
        except Exception as e:
            print(f"Error getting collection stats: {e}")
        finally:
            conn.close()

    def test_search(self, num_queries: int = 10):
        """Test vector similarity search"""
        print(f"\nTesting vector search with {num_queries} queries...")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                total_time = 0
                for i in range(num_queries):
                    # Generate random query vector
                    query_vector = self.generate_random_vector()
                    
                    start_time = time.time()
                    cur.execute("""
                        SELECT vector_id, text_content, metadata, 
                               1 - (embedding <=> %s::vector) as similarity
                        FROM vector_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, query_vector))
                    
                    results = cur.fetchall()
                    search_time = time.time() - start_time
                    total_time += search_time
                    
                    if i == 0:  # Show first result as example
                        print(f"Sample search result: {len(results)} matches, top similarity: {results[0][3]:.4f}")
                
                avg_search_time = total_time / num_queries
                print(f"Average search time: {avg_search_time:.4f}s")
                print(f"Search QPS: {1/avg_search_time:.2f}")
                
        except Exception as e:
            print(f"Error during search test: {e}")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Populate PostgreSQL vector database with test data")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", default="vectordb", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--records", type=int, default=10_000_000, help="Number of records to insert")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--vector-dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--test-search", action="store_true", help="Run search test after population")
    
    args = parser.parse_args()
    
    # Create populator instance
    populator = PostgreSQLVectorPopulator(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
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
        
        # Test search if requested
        if args.test_search:
            populator.test_search()
        
    except KeyboardInterrupt:
        print("\nPopulation interrupted by user")
    except Exception as e:
        print(f"Error during population: {e}")
        raise


if __name__ == "__main__":
    main()

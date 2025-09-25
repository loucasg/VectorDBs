#!/usr/bin/env python3
"""
Populate TimescaleDB with vector embeddings for benchmarking
Updated to work with pgvectorscale extension and DiskANN indexes
"""

import time
import psycopg2
import numpy as np
import argparse
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

class TimescaleDBPopulator:
    def __init__(self, host="localhost", port=5433, user="postgres", password="postgres", dbname="vectordb"):
        self.connection_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": dbname
        }
        self.vector_dim = 1024
        
    def connect(self):
        """Connect to TimescaleDB"""
        try:
            self.conn = psycopg2.connect(**self.connection_config)
            self.conn.autocommit = True
            print(f"✅ Connected to TimescaleDB at {self.connection_config['host']}:{self.connection_config['port']}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to TimescaleDB: {e}")
            return False
    
    def check_extensions(self):
        """Check if required extensions are available"""
        try:
            with self.conn.cursor() as cur:
                # Check for TimescaleDB extension
                cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');")
                has_timescaledb = cur.fetchone()[0]
                
                # Check for pgvector extension
                cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
                has_vector = cur.fetchone()[0]
                
                # Check for pgvectorscale extension
                cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vectorscale');")
                has_vectorscale = cur.fetchone()[0]
                
                print(f"📊 Extensions status:")
                print(f"   TimescaleDB: {'✅' if has_timescaledb else '❌'}")
                print(f"   pgvector: {'✅' if has_vector else '❌'}")
                print(f"   pgvectorscale: {'✅' if has_vectorscale else '❌'}")
                
                if not (has_timescaledb and has_vector and has_vectorscale):
                    print("❌ Required extensions not found. Please run init-timescaledb.sql first.")
                    return False
                
                return True
                
        except Exception as e:
            print(f"❌ Error checking extensions: {e}")
            return False
    
    def check_table_exists(self):
        """Check if the vector_embeddings table exists"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'vector_embeddings'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    print("✅ TimescaleDB table exists, will append new records")
                    return True
                else:
                    print("❌ Table 'vector_embeddings' not found. Please run init-timescaledb.sql first.")
                    return False
                    
        except Exception as e:
            print(f"❌ Error checking table: {e}")
            return False
    
    def get_diskann_stats(self):
        """Get DiskANN index statistics"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM get_diskann_index_stats();")
                stats = cur.fetchall()
                
                if stats:
                    print("📊 DiskANN Index Statistics:")
                    for stat in stats:
                        print(f"   {stat[0]}: {stat[1]} ({stat[2]})")
                else:
                    print("⚠️  No DiskANN indexes found")
                    
        except Exception as e:
            print(f"⚠️  Could not retrieve DiskANN stats: {e}")
    
    def optimize_diskann(self):
        """Optimize DiskANN for better performance"""
        try:
            with self.conn.cursor() as cur:
                # Optimize DiskANN query performance
                cur.execute("SELECT optimize_diskann_query(400, true);")
                result = cur.fetchone()[0]
                print(f"🔧 DiskANN optimization: {result}")
                
        except Exception as e:
            print(f"⚠️  Could not optimize DiskANN: {e}")
    
    def generate_vector(self):
        """Generate a random vector for testing"""
        return np.random.rand(self.vector_dim).astype(np.float32).tolist()
    
    def get_connection(self):
        """Get a new database connection for multi-threading"""
        return psycopg2.connect(**self.connection_config)

    def insert_batch(self, batch_data, start_vector_id):
        """Insert a batch of data into TimescaleDB (thread-safe version)"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare data for batch insert
                data = []
                for i, (vector, text, metadata) in enumerate(batch_data):
                    data.append((start_vector_id + i + 1, vector, text, json.dumps(metadata)))
                
                # Batch insert using the new schema
                cur.executemany("""
                    INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                    VALUES (%s, %s::vector, %s, %s)
                """, data)
                
                conn.commit()
                return True, len(batch_data)
                
        except Exception as e:
            print(f"❌ Error inserting batch: {e}")
            conn.rollback()
            return False, 0
        finally:
            conn.close()
    
    def get_count(self):
        """Get the current count of records"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                return cur.fetchone()[0]
        except Exception as e:
            print(f"❌ Error getting count: {e}")
            return 0
    
    def get_collection_stats(self):
        """Get collection statistics using the pgvectorscale function"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM get_collection_stats();")
                stats = cur.fetchone()
                
                if stats:
                    print(f"📊 Collection Statistics:")
                    print(f"   Total points: {stats[0]:,}")
                    print(f"   Vector dimensions: {stats[1]}")
                    print(f"   Avg metadata size: {stats[2]:.2f} bytes")
                    print(f"   Created at range: {stats[3]}")
                    print(f"   Hypertable info: {stats[4]}")
                    print(f"   DiskANN indexes: {stats[5]}")
                    print(f"   Index sizes: {stats[6]}")
                    
        except Exception as e:
            print(f"⚠️  Could not retrieve collection stats: {e}")
    
    def generate_batch_data(self, batch_size, start_id):
        """Generate batch data for insertion"""
        batch_data = []
        for i in range(batch_size):
            vector = self.generate_vector()
            text = f"TimescaleDB+pgvectorscale document {start_id + i + 1}"
            metadata = {
                "source": "timescaledb_vectorscale", 
                "batch": start_id // batch_size, 
                "index": i, 
                "extension": "pgvectorscale"
            }
            batch_data.append((vector, text, metadata))
        return batch_data

    def populate(self, num_records=100000, batch_size=1000, max_workers=4):
        """Populate the database with vector embeddings using multi-threading"""
        print(f"\n{'='*50}")
        print(f"POPULATING TIMESCALEDB DATABASE")
        print(f"{'='*50}")
        print(f"Database: {self.connection_config['dbname']}")
        print(f"Records to insert: {num_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {batch_size}")
        print(f"Max workers: {max_workers}")
        print(f"Extension: pgvectorscale + DiskANN")
        
        if not self.connect():
            return False
        
        # Check extensions and table
        if not self.check_extensions():
            return False
            
        if not self.check_table_exists():
            return False
        
        # Get initial collection stats
        print("\n📊 Initial collection state:")
        self.get_collection_stats()
        
        # Check if data already exists and get current max vector_id
        current_count = self.get_count()
        current_max_id = 0
        if current_count > 0:
            print(f"\n📊 TimescaleDB already contains {current_count:,} records")
            # Get current max vector_id
            with self.conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                current_max_id = cur.fetchone()[0]
            print(f"📊 Current max vector_id: {current_max_id:,}")
            print("📝 Will append new records to existing data")
        
        # Optimize DiskANN before bulk insert
        print("\n🔧 Optimizing DiskANN for bulk insert...")
        self.optimize_diskann()
        
        start_time = time.time()
        total_batches = (num_records + batch_size - 1) // batch_size
        
        # Create progress bar
        pbar = tqdm(total=num_records, desc="Inserting records", unit="records")
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        successful_batches = 0
        failed_batches = 0
        total_inserted = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for batch_idx in range(total_batches):
                batch_start_id = current_max_id + (batch_idx * batch_size)
                current_batch_size = min(batch_size, num_records - (batch_idx * batch_size))
                
                # Generate batch data
                batch_data = self.generate_batch_data(current_batch_size, batch_start_id)
                
                # Submit batch for insertion
                future = executor.submit(self.insert_batch, batch_data, batch_start_id)
                futures.append(future)
                
                # Process completed futures
                if len(futures) >= max_workers * 2:  # Keep some futures in flight
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            success, inserted = future.result()
                            if success:
                                successful_batches += 1
                                total_inserted += inserted
                            else:
                                failed_batches += 1
                            completed_futures.append(future)
                    
                    # Remove completed futures
                    for future in completed_futures:
                        futures.remove(future)
                        pbar.update(batch_size)
            
            # Wait for remaining futures
            for future in futures:
                success, inserted = future.result()
                if success:
                    successful_batches += 1
                    total_inserted += inserted
                else:
                    failed_batches += 1
                pbar.update(batch_size)
        
        pbar.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Get final count
        final_count = self.get_count()
        
        print(f"\n{'='*50}")
        print(f"POPULATION COMPLETED")
        print(f"{'='*50}")
        print(f"Records added: {total_inserted:,}")
        print(f"Previous count: {current_count:,}")
        print(f"Final count: {final_count:,}")
        print(f"Successful batches: {successful_batches:,}")
        print(f"Failed batches: {failed_batches:,}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Records per second: {total_inserted / duration:,.0f}")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory: {final_memory:.2f} MB")
        
        # Get final collection stats
        print(f"\n{'='*50}")
        print(f"COLLECTION STATISTICS")
        print(f"{'='*50}")
        print(f"Database: {self.connection_config['dbname']}")
        print(f"Total points: {final_count:,}")
        print(f"Vector dimensions: {self.vector_dim}")
        print(f"Extension: pgvectorscale + DiskANN")
        print(f"Index type: DiskANN")
        print(f"Distance metric: COSINE")
        
        # Get DiskANN index stats
        print("\n📊 DiskANN Index Statistics:")
        self.get_diskann_stats()
        
        # Final DiskANN optimization
        print("\n🔧 Final DiskANN optimization...")
        self.optimize_diskann()
        
        self.conn.close()
        return True

def main():
    parser = argparse.ArgumentParser(description="Populate TimescaleDB with vector embeddings using pgvectorscale")
    parser.add_argument("--records", type=int, default=100000, help="Number of records to insert")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--host", default="localhost", help="TimescaleDB host")
    parser.add_argument("--port", type=int, default=5433, help="TimescaleDB port")
    parser.add_argument("--user", default="postgres", help="TimescaleDB user")
    parser.add_argument("--password", default="postgres", help="TimescaleDB password")
    parser.add_argument("--dbname", default="vectordb", help="TimescaleDB database name")
    
    args = parser.parse_args()
    
    populator = TimescaleDBPopulator(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )
    
    populator.populate(num_records=args.records, batch_size=args.batch_size, max_workers=args.workers)

if __name__ == "__main__":
    main()

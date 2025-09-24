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
    
    def insert_batch(self, vectors, text_contents, metadata_list, start_vector_id=None):
        """Insert a batch of data into TimescaleDB"""
        try:
            with self.conn.cursor() as cur:
                # Get current max vector_id if not provided
                if start_vector_id is None:
                    cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                    start_vector_id = cur.fetchone()[0]
                
                # Prepare data for batch insert
                data = []
                for i, (vector, text, metadata) in enumerate(zip(vectors, text_contents, metadata_list)):
                    data.append((start_vector_id + i + 1, vector, text, json.dumps(metadata)))
                
                # Batch insert using the new schema
                cur.executemany("""
                    INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                    VALUES (%s, %s::vector, %s, %s)
                """, data)
                
                return True, len(vectors)
                
        except Exception as e:
            print(f"❌ Error inserting batch: {e}")
            return False, 0
    
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
    
    def populate(self, num_records=100000, batch_size=1000):
        """Populate the database with vector embeddings"""
        print(f"\n{'='*50}")
        print(f"POPULATING TIMESCALEDB DATABASE")
        print(f"{'='*50}")
        print(f"Database: {self.connection_config['dbname']}")
        print(f"Records to insert: {num_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {batch_size}")
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
        total_inserted = 0
        current_id = current_max_id
        
        # Generate and insert data in batches
        print(f"\n📝 Inserting {num_records:,} records in batches of {batch_size:,}...")
        for batch_start in tqdm(range(0, num_records, batch_size), desc="Inserting batches"):
            batch_end = min(batch_start + batch_size, num_records)
            batch_size_actual = batch_end - batch_start
            
            # Generate batch data
            vectors = [self.generate_vector() for _ in range(batch_size_actual)]
            text_contents = [f"TimescaleDB+pgvectorscale document {current_id + i + 1}" for i in range(batch_size_actual)]
            metadata_list = [{"source": "timescaledb_vectorscale", "batch": batch_start // batch_size, "index": i, "extension": "pgvectorscale"} 
                           for i in range(batch_size_actual)]
            
            # Insert batch
            success, inserted = self.insert_batch(vectors, text_contents, metadata_list, current_id)
            if success:
                total_inserted += inserted
                current_id += inserted  # Update current_id for next batch
            else:
                print(f"❌ Failed to insert batch starting at {batch_start}")
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"POPULATION COMPLETED")
        print(f"{'='*50}")
        print(f"Records inserted: {total_inserted:,}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Records per second: {total_inserted/duration:.0f}")
        
        # Verify final count and get updated stats
        final_count = self.get_count()
        print(f"Total records in TimescaleDB: {final_count:,}")
        
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
    
    populator.populate(num_records=args.records, batch_size=args.batch_size)

if __name__ == "__main__":
    main()

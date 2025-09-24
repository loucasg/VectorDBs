#!/usr/bin/env python3
"""
Populate TimescaleDB with vector embeddings for benchmarking
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
    
    def create_table(self):
        """Create the vector embeddings table with TimescaleDB hypertable"""
        try:
            with self.conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'vector_embeddings'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    # Create the table
                    cur.execute("""
                        CREATE TABLE vector_embeddings (
                            vector_id BIGINT,
                            embedding VECTOR(1024),
                            text_content TEXT,
                            metadata JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            PRIMARY KEY (vector_id, created_at)
                        );
                    """)
                    
                    # Create TimescaleDB hypertable
                    cur.execute("""
                        SELECT create_hypertable('vector_embeddings', 'created_at');
                    """)
                else:
                    print("✅ TimescaleDB table already exists, will append new records")
                
                # Create vector similarity index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS vector_embeddings_embedding_idx 
                    ON vector_embeddings USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """)
                
                # Create metadata index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS vector_embeddings_metadata_idx 
                    ON vector_embeddings USING gin (metadata);
                """)
                
                print("✅ TimescaleDB table and indexes created successfully")
                return True
                
        except Exception as e:
            print(f"❌ Error creating TimescaleDB table: {e}")
            return False
    
    def generate_vector(self):
        """Generate a random vector for testing"""
        return np.random.rand(self.vector_dim).astype(np.float32).tolist()
    
    def insert_batch(self, vectors, text_contents, metadata_list, start_id=None):
        """Insert a batch of data into TimescaleDB"""
        try:
            with self.conn.cursor() as cur:
                # Get current max ID if not provided
                if start_id is None:
                    cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                    start_id = cur.fetchone()[0]
                
                # Prepare data for batch insert
                data = []
                for i, (vector, text, metadata) in enumerate(zip(vectors, text_contents, metadata_list)):
                    data.append((start_id + i + 1, vector, text, json.dumps(metadata)))
                
                # Batch insert
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
    
    def populate(self, num_records=100000, batch_size=1000):
        """Populate the database with vector embeddings"""
        print(f"🚀 Starting TimescaleDB population with {num_records:,} records...")
        
        if not self.connect():
            return False
            
        if not self.create_table():
            return False
        
        # Check if data already exists and get current max ID
        current_count = self.get_count()
        current_max_id = 0
        if current_count > 0:
            print(f"📊 TimescaleDB already contains {current_count:,} records")
            # Get current max ID
            with self.conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                current_max_id = cur.fetchone()[0]
            print(f"📊 Current max ID: {current_max_id:,}")
            print("📝 Will append new records to existing data")
        
        start_time = time.time()
        total_inserted = 0
        current_id = current_max_id
        
        # Generate and insert data in batches
        for batch_start in tqdm(range(0, num_records, batch_size), desc="Inserting batches"):
            batch_end = min(batch_start + batch_size, num_records)
            batch_size_actual = batch_end - batch_start
            
            # Generate batch data
            vectors = [self.generate_vector() for _ in range(batch_size_actual)]
            text_contents = [f"TimescaleDB document {current_id + i + 1}" for i in range(batch_size_actual)]
            metadata_list = [{"source": "timescaledb", "batch": batch_start // batch_size, "index": i} 
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
        
        print(f"\n✅ TimescaleDB population completed!")
        print(f"📊 Records inserted: {total_inserted:,}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"🚀 Insert rate: {total_inserted/duration:.0f} records/second")
        
        # Verify final count
        final_count = self.get_count()
        print(f"📈 Total records in TimescaleDB: {final_count:,}")
        
        self.conn.close()
        return True

def main():
    parser = argparse.ArgumentParser(description="Populate TimescaleDB with vector embeddings")
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

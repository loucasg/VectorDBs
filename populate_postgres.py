#!/usr/bin/env python3
"""
Optimized PostgreSQL Vector Database Population Script
Updated to match optimized PostgreSQL schema with partitioning and IVFFlat indexes
"""

import time
import random
import numpy as np
import json
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
import threading
import queue


class OptimizedPostgreSQLVectorPopulator:
    def __init__(self, host: str = "localhost", port: int = 5432,
                 database: str = "vectordb", user: str = "postgres",
                 password: str = "postgres"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.vector_dim = 1024  # Default, will be detected from schema
        self.batch_size = 5000  # Larger batches for IVFFlat optimization
        self._vector_dim_detected = False

    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )

    def detect_schema_and_optimize(self):
        """Detect existing schema and apply optimizations"""
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
                
                if not table_exists:
                    raise Exception("Table 'vector_embeddings' not found! Please run the optimized PostgreSQL init script first.")
                
                # Detect vector dimensions from existing data or schema
                cur.execute("""
                    SELECT column_name, data_type, udt_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'vector_embeddings' 
                    AND column_name = 'embedding';
                """)
                column_info = cur.fetchone()
                
                if column_info and 'vector' in str(column_info):
                    # Try to detect dimensions from existing data
                    cur.execute("SELECT vector_dims(embedding) FROM vector_embeddings LIMIT 1;")
                    result = cur.fetchone()
                    if result and result[0]:
                        self.vector_dim = result[0]
                        self._vector_dim_detected = True
                        print(f"Detected vector dimensions: {self.vector_dim}")
                    else:
                        print(f"Using default vector dimensions: {self.vector_dim}")
                
                # Check if partitions exist
                cur.execute("""
                    SELECT COUNT(*) FROM pg_inherits 
                    WHERE inhparent = 'vector_embeddings'::regclass;
                """)
                partition_count = cur.fetchone()[0]
                print(f"Found {partition_count} partitions")
                
                # Apply session optimizations using the schema's optimization function
                try:
                    cur.execute("SELECT optimize_for_vector_queries();")
                    result = cur.fetchone()[0]
                    print(f"Applied optimizations: {result}")
                except Exception as e:
                    print(f"Note: Could not apply schema optimizations: {e}")
                    # Apply manual optimizations
                    cur.execute("SET work_mem = '1GB';")
                    cur.execute("SET maintenance_work_mem = '4GB';")
                    cur.execute("SET ivfflat.probes = '100';")
                
                conn.commit()
                
        except Exception as e:
            print(f"Error detecting schema: {e}")
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

    def generate_chunk_data(self, chunk_id: int, base_time: datetime) -> Dict[str, Any]:
        """Generate sample chunk data with timestamps for partitioning"""
        # Add some randomness to timestamps for partition distribution
        time_offset = timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        record_time = base_time + time_offset

        return {
            "id": chunk_id,
            "text": f"Optimized PostgreSQL document chunk {chunk_id} created at {record_time.isoformat()}. " + 
                   "This contains sample content for vector similarity testing. " * random.randint(1, 3),
            "metadata": {
                "source": f"document_{chunk_id % 10000}",
                "page": random.randint(1, 500),
                "section": random.choice(["abstract", "introduction", "methodology", "results", "conclusion", "references"]),
                "timestamp": int(record_time.timestamp()),
                "category": random.choice(["research", "technical", "business", "legal", "scientific", "medical"]),
                "confidence": round(random.uniform(0.6, 1.0), 3),
                "author_id": f"author_{random.randint(1, 1000)}",
                "department": random.choice(["engineering", "research", "sales", "marketing", "legal"]),
                "priority": random.choice(["low", "medium", "high"]),
                "language": random.choice(["en", "es", "fr", "de", "zh"])
            },
            "created_at": record_time
        }

    def get_next_available_id(self, count: int = 1) -> int:
        """Get the next available ID range atomically"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Use a more robust approach for high concurrency
                cur.execute("""
                    INSERT INTO vector_id_counter (id, last_id)
                    VALUES (1, (SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings))
                    ON CONFLICT (id) DO UPDATE SET 
                    last_id = vector_id_counter.last_id + %s
                    RETURNING last_id + 1;
                """, (count,))

                result = cur.fetchone()
                next_id = result[0] if result else 1
                conn.commit()
                return next_id
        except Exception as e:
            print(f"Error getting next available ID: {e}")
            # Fallback with timestamp
            return int(time.time() * 1000) % 1000000000
        finally:
            conn.close()

    def bulk_insert_optimized(self, batch_data: List[Dict[str, Any]]) -> int:
        """Use the schema's bulk insert function if available, otherwise optimized insert"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare arrays for bulk function
                vector_ids = []
                embeddings = []
                text_contents = []
                metadatas = []

                for record in batch_data:
                    vector = self.generate_random_vector()
                    chunk_data = record['chunk_data']

                    vector_ids.append(record['id'])
                    embeddings.append(vector)
                    text_contents.append(chunk_data['text'])
                    metadatas.append(chunk_data['metadata'])

                # Try to use the schema's optimized bulk insert function
                try:
                    # Convert embeddings to proper format
                    vector_arrays = [f"[{','.join(map(str, emb))}]" for emb in embeddings]
                    cur.execute("""
                        SELECT bulk_insert_vectors_optimized(%s::BIGINT[], %s::VECTOR[], %s::TEXT[], %s::JSONB[]);
                    """, (vector_ids, vector_arrays, text_contents, [json.dumps(m) for m in metadatas]))
                    
                    result = cur.fetchone()[0]
                    conn.commit()
                    return result
                    
                except Exception as schema_error:
                    # Fallback to manual optimized insert
                    print(f"Schema bulk function not available, using fallback: {schema_error}")
                    conn.rollback()
                    
                    # Manual optimized batch insert
                    cur.execute("SET maintenance_work_mem = '4GB';")
                    
                    # Note: Cannot disable autovacuum on partitioned tables
                    
                    batch_tuples = []
                    for i, record_id in enumerate(vector_ids):
                        batch_tuples.append((
                            record_id,
                            embeddings[i],
                            text_contents[i],
                            json.dumps(metadatas[i]),
                            batch_data[i]['chunk_data']['created_at']
                        ))

                    execute_values(
                        cur,
                        """
                        INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata, created_at)
                        VALUES %s
                        ON CONFLICT (vector_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            text_content = EXCLUDED.text_content,
                            metadata = EXCLUDED.metadata,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        batch_tuples,
                        page_size=self.batch_size
                    )
                    
                    # Note: Autovacuum is managed by PostgreSQL for partitioned tables
                    
                    conn.commit()
                    return len(batch_tuples)

        except Exception as e:
            print(f"Error in bulk insert: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def populate_database(self, total_records: int = 1_000_000, time_spread_days: int = 365):
        """Populate the database with optimized batching"""
        print(f"\n{'='*60}")
        print(f"POPULATING OPTIMIZED POSTGRESQL DATABASE")
        print(f"{'='*60}")
        print(f"Database: {self.database}")
        print(f"Records to insert: {total_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Time spread: {time_spread_days} days")

        # Detect schema and optimize
        self.detect_schema_and_optimize()

        # Get current count
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                current_count = cur.fetchone()[0]
                print(f"Current records in table: {current_count:,}")
        except Exception as e:
            print(f"Error getting current count: {e}")
            current_count = 0
        finally:
            conn.close()

        # Get next available ID range
        start_id = self.get_next_available_id(total_records)
        print(f"Starting from ID: {start_id}")
        print(f"Will add {total_records:,} new records")

        # Create time range for partition distribution
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_spread_days)
        print(f"Time range: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")

        start_population_time = time.time()
        total_batches = (total_records + self.batch_size - 1) // self.batch_size

        # Create progress bar
        pbar = tqdm(total=total_records, desc="Inserting optimized records", unit="records")

        records_inserted = 0
        successful_batches = 0
        failed_batches = 0

        # Process batches
        for batch_idx in range(total_batches):
            batch_start_id = start_id + (batch_idx * self.batch_size)
            current_batch_size = min(self.batch_size, total_records - (batch_idx * self.batch_size))

            # Create batch with time distribution for partitioning
            batch_data = []
            batch_base_time = start_time + timedelta(
                days=int((batch_idx / total_batches) * time_spread_days)
            )

            for i in range(current_batch_size):
                record_id = batch_start_id + i
                chunk_data = self.generate_chunk_data(record_id, batch_base_time)
                batch_data.append({
                    "id": record_id,
                    "chunk_data": chunk_data
                })

            # Insert batch
            result = self.bulk_insert_optimized(batch_data)
            if result > 0:
                successful_batches += 1
                records_inserted += result
                pbar.update(result)
            else:
                failed_batches += 1

        pbar.close()

        # Final statistics
        end_time = time.time()
        duration = end_time - start_population_time

        # Get final count
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                final_count = cur.fetchone()[0]
        except Exception as e:
            final_count = current_count + records_inserted
        finally:
            conn.close()

        print(f"\n{'='*60}")
        print(f"OPTIMIZED POSTGRESQL POPULATION COMPLETED")
        print(f"{'='*60}")
        print(f"Records added: {records_inserted:,}")
        print(f"Previous count: {current_count:,}")
        print(f"Final count: {final_count:,}")
        print(f"Successful batches: {successful_batches:,}")
        print(f"Failed batches: {failed_batches:,}")
        print(f"Duration: {duration:.2f} seconds")
        if duration > 0:
            print(f"Records per second: {records_inserted / duration:,.0f}")

    def get_collection_stats(self):
        """Get collection statistics using schema function if available"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Try to use schema's advanced stats function
                try:
                    cur.execute("SELECT * FROM get_collection_stats_advanced();")
                    stats = cur.fetchone()
                    
                    print(f"\n{'='*60}")
                    print(f"OPTIMIZED POSTGRESQL STATISTICS")
                    print(f"{'='*60}")
                    print(f"Total points: {stats['total_points']:,}")
                    print(f"Partition count: {stats['partition_count']}")
                    print(f"Vector dimensions: {stats['vector_dimensions']}")
                    print(f"Avg vectors per partition: {stats['avg_vectors_per_partition']:,}")
                    print(f"Table size: {stats['table_size']}")
                    print(f"Largest partition: {stats['largest_partition']}")
                    
                except Exception:
                    # Fallback to basic stats
                    cur.execute("SELECT COUNT(*) as total_points FROM vector_embeddings;")
                    total_points = cur.fetchone()['total_points']
                    
                    print(f"\n{'='*60}")
                    print(f"POSTGRESQL STATISTICS")
                    print(f"{'='*60}")
                    print(f"Total points: {total_points:,}")
                    print(f"Vector dimensions: {self.vector_dim}")

        except Exception as e:
            print(f"Error getting collection stats: {e}")
        finally:
            conn.close()

    def test_vector_search(self, num_queries: int = 5):
        """Test vector search performance"""
        print(f"\nTesting PostgreSQL vector search ({num_queries} queries)...")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Apply optimizations
                cur.execute("SET ivfflat.probes = 100;")
                cur.execute("SET work_mem = '1GB';")
                
                total_time = 0
                
                for i in range(num_queries):
                    query_vector = self.generate_random_vector()
                    
                    start_time = time.time()
                    
                    # Try optimized search function first
                    try:
                        cur.execute("""
                            SELECT * FROM search_similar_vectors_optimized(%s, 10, 0.0, false);
                        """, (query_vector,))
                    except Exception:
                        # Fallback to direct query
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
                    
                    if i == 0:
                        print(f"Sample results: {len(results)} matches, top similarity: {results[0][3]:.4f}")
                
                avg_time = total_time / num_queries
                print(f"Average search time: {avg_time:.4f}s")
                print(f"Queries per second: {1/avg_time:.2f}")
                
        except Exception as e:
            print(f"Error during search test: {e}")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Populate optimized PostgreSQL vector database")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", default="vectordb", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--records", type=int, default=1_000_000, help="Number of records to insert")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for inserts")
    parser.add_argument("--time-spread-days", type=int, default=365, help="Spread data across N days")
    parser.add_argument("--show-stats", action="store_true", help="Show collection stats after population")
    parser.add_argument("--test-search", action="store_true", help="Test search performance")

    args = parser.parse_args()

    populator = OptimizedPostgreSQLVectorPopulator(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )
    populator.batch_size = args.batch_size

    try:
        populator.populate_database(
            total_records=args.records,
            time_spread_days=args.time_spread_days
        )

        if args.show_stats:
            populator.get_collection_stats()
            
        if args.test_search:
            populator.test_vector_search()

    except KeyboardInterrupt:
        print("\nPopulation interrupted by user")
    except Exception as e:
        print(f"Error during population: {e}")
        raise


if __name__ == "__main__":
    main()
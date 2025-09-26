#!/usr/bin/env python3
"""
Optimized TimescaleDB Vector Database Population Script
Updated to match the hybrid/optimized TimescaleDB schema with pgvectorscale
"""

import time
import random
import numpy as np
import json
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import argparse
from tqdm import tqdm
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os


class OptimizedTimescaleDBVectorPopulator:
    def __init__(self, host: str = "localhost", port: int = 5433,
                 database: str = "vectordb", user: str = "postgres", 
                 password: str = "postgres", schema: str = "public"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema
        self.table_name = "vector_embeddings_ts"  # Matches optimized_timescaledb.sql
        self.vector_dim = 1024  # Will be auto-detected
        self.batch_size = 500  # Optimized for TimescaleDB chunks with concurrent workers
        self._schema_detected = False

    def get_connection(self):
        """Get a database connection with schema path"""
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        # Set search path
        with conn.cursor() as cur:
            cur.execute(f"SET search_path = {self.schema}, public;")
        return conn

    def detect_schema_and_optimize(self):
        """Detect existing schema and apply optimizations"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Check if schema exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.schemata 
                        WHERE schema_name = %s
                    );
                """, (self.schema,))
                schema_exists = cur.fetchone()[0]
                
                if not schema_exists:
                    print(f"Schema '{self.schema}' not found! Creating it...")
                    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema};")
                    conn.commit()

                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = 'vector_embeddings_ts'
                    );
                """, (self.schema,))
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    raise Exception(f"Table '{self.table_name}' not found! Please run the optimized TimescaleDB init script first.")

                # Check if it's a hypertable
                cur.execute("""
                    SELECT hypertable_name, primary_dimension, num_chunks 
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = %s AND hypertable_name = 'vector_embeddings_ts';
                """, (self.schema,))
                hypertable_info = cur.fetchone()
                
                if not hypertable_info:
                    raise Exception(f"Table '{self.table_name}' exists but is not a hypertable! Please run the optimized init script.")
                
                print(f"Found hypertable: {hypertable_info[0]}")
                print(f"Time column: {hypertable_info[1]}")
                print(f"Partitioning column: {hypertable_info[2] or 'None'}")

                # Detect vector dimensions
                cur.execute(f"""
                    SELECT vector_dims(embedding) 
                    FROM {self.table_name} 
                    WHERE embedding IS NOT NULL 
                    LIMIT 1;
                """)
                result = cur.fetchone()
                if result and result[0]:
                    self.vector_dim = result[0]
                    print(f"Detected vector dimensions: {self.vector_dim}")
                else:
                    # Check column definition
                    cur.execute(f"""
                        SELECT column_name, data_type, udt_name 
                        FROM information_schema.columns 
                        WHERE table_schema = %s AND table_name = 'embeddings_ts' 
                        AND column_name = 'embedding';
                    """, (self.schema,))
                    column_info = cur.fetchone()
                    if column_info:
                        print(f"Vector column found: {column_info}")
                        # For VECTOR(1024), extract the dimension
                        if 'vector' in str(column_info).lower() and '1024' in str(column_info):
                            self.vector_dim = 1024
                    print(f"Using default vector dimensions: {self.vector_dim}")

                # Apply schema optimizations if available
                try:
                    cur.execute("SELECT optimize_for_vector_operations();")
                    result = cur.fetchone()[0]
                    print(f"Applied optimizations: {result}")
                except Exception as e:
                    print(f"Schema optimization function not available, applying manual settings: {e}")
                    # Manual optimizations
                    cur.execute("SET work_mem = '512MB';")
                    cur.execute("SET maintenance_work_mem = '2GB';")
                    cur.execute("SET vectorscale.query_rescore = '200';")

                self._schema_detected = True
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

    def generate_chunk_data_with_timestamp(self, chunk_id: int, base_time: datetime) -> Dict[str, Any]:
        """Generate sample chunk data with realistic timestamps for time-series"""
        # Create realistic time distribution within chunks
        time_offset = timedelta(
            minutes=random.randint(0, 30),  # Within 30 minutes for chunk locality
            seconds=random.randint(0, 59)
        )
        record_time = base_time + time_offset

        return {
            "id": chunk_id,
            "text": f"TimescaleDB hypertable chunk {chunk_id} at {record_time.isoformat()}. " +
                   "This is optimized time-series vector data with DiskANN indexing. " * random.randint(1, 2),
            "metadata": {
                "source": f"timeseries_doc_{chunk_id % 5000}",
                "chunk_id": chunk_id,
                "page": random.randint(1, 200),
                "section": random.choice(["data", "analysis", "summary", "appendix", "methodology"]),
                "timestamp": int(record_time.timestamp()),
                "category": random.choice(["timeseries", "analytics", "monitoring", "alerts", "metrics"]),
                "confidence": round(random.uniform(0.7, 1.0), 3),
                "sensor_type": random.choice(["temperature", "pressure", "humidity", "motion", "light"]),
                "location": random.choice(["datacenter_1", "datacenter_2", "edge_site_a", "edge_site_b"]),
                "priority": random.choice(["normal", "high", "critical"]),
                "device_id": f"device_{random.randint(1, 500)}",
                "batch_id": chunk_id // self.batch_size
            },
            "created_at": record_time
        }

    def get_next_available_id(self, count: int = 1) -> int:
        """Get the next available ID range"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COALESCE(MAX(vector_id), 0) + 1 FROM {self.table_name};")
                next_id = cur.fetchone()[0]
                return next_id
        except Exception as e:
            print(f"Error getting next available ID: {e}")
            return int(time.time() * 1000) % 1000000000
        finally:
            conn.close()

    def bulk_insert_optimized(self, batch_data: List[Dict[str, Any]], worker_id: int = 0) -> int:
        """Use schema's bulk insert function or optimized fallback"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Set TimescaleDB optimizations
                cur.execute("SET work_mem = '256MB';")
                cur.execute("SET maintenance_work_mem = '1GB';")
                cur.execute("SET lock_timeout = '30s';")

                # Prepare data arrays for bulk function
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

                # Apply TimescaleDB optimizations
                cur.execute("SELECT optimize_for_vector_operations();")
                
                # Try schema's bulk insert function first
                try:
                    cur.execute("""
                        SELECT bulk_insert_vectors(%s, %s, %s, %s);
                    """, (vector_ids, embeddings, text_contents, [json.dumps(m) for m in metadatas]))
                    
                    result = cur.fetchone()[0]
                    conn.commit()
                    return result

                except Exception as schema_error:
                    # Fallback to manual optimized insert for TimescaleDB
                    conn.rollback()
                    
                    # Use VALUES clause for better performance with TimescaleDB
                    batch_tuples = []
                    for i, record_id in enumerate(vector_ids):
                        batch_tuples.append((
                            record_id,
                            embeddings[i],
                            text_contents[i],
                            json.dumps(metadatas[i]),
                            batch_data[i]['chunk_data']['created_at']
                        ))

                    # Use psycopg2's execute_values for optimal performance
                    from psycopg2.extras import execute_values
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {self.table_name} (vector_id, embedding, text_content, metadata, created_at)
                        VALUES %s
                        """,
                        batch_tuples,
                        page_size=self.batch_size
                    )
                    
                    conn.commit()
                    return len(batch_tuples)

        except psycopg2.errors.DeadlockDetected:
            # Handle TimescaleDB deadlocks with exponential backoff
            conn.rollback()
            backoff_time = min(0.1 * (2 ** (worker_id % 4)), 2.0)  # Max 2 seconds
            time.sleep(backoff_time)
            return 0
        except Exception as e:
            print(f"Worker {worker_id} - Error in bulk insert: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def populate_database(self, total_records: int = 10_000_000, max_workers: int = 2,
                         time_spread_days: int = 30):
        """Populate the TimescaleDB with optimized time-series data"""
        print(f"\n{'='*70}")
        print(f"POPULATING OPTIMIZED TIMESCALEDB DATABASE")
        print(f"{'='*70}")
        print(f"Database: {self.database}")
        print(f"Schema: {self.schema}")
        print(f"Table: {self.table_name}")
        print(f"Records to insert: {total_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max workers: {max_workers}")
        print(f"Time spread: {time_spread_days} days")

        # Detect schema and optimize
        self.detect_schema_and_optimize()

        # Get current count
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                current_count = cur.fetchone()[0]
                print(f"Current records in hypertable: {current_count:,}")
                
                # Get chunk information
                cur.execute("""
                    SELECT COUNT(*) as chunk_count
                    FROM timescaledb_information.chunks
                    WHERE hypertable_schema = %s AND hypertable_name = 'embeddings_ts';
                """, (self.schema,))
                chunk_info = cur.fetchone()
                print(f"Current chunks: {chunk_info[0] if chunk_info else 0}")
                
        except Exception as e:
            print(f"Error getting current count: {e}")
            current_count = 0
        finally:
            conn.close()

        # Get next available ID
        start_id = self.get_next_available_id()
        print(f"Starting from ID: {start_id}")
        print(f"Will add {total_records:,} new records")

        # Create time range for realistic time-series distribution
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_spread_days)
        print(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")

        start_population_time = time.time()
        total_batches = (total_records + self.batch_size - 1) // self.batch_size

        # Track performance
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create progress bar
        pbar = tqdm(total=total_records, desc="Inserting TimescaleDB records", unit="records")

        successful_batches = 0
        failed_batches = 0
        records_inserted = 0

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for batch_idx in range(total_batches):
                batch_start_id = start_id + (batch_idx * self.batch_size)
                current_batch_size = min(self.batch_size, total_records - (batch_idx * self.batch_size))

                # Create batch data with time locality for TimescaleDB chunks
                batch_data = []
                # Calculate time for this batch to maintain chunk locality
                batch_time_offset = (batch_idx / total_batches) * time_spread_days * 24 * 3600
                batch_base_time = start_time + timedelta(seconds=int(batch_time_offset))

                for i in range(current_batch_size):
                    record_id = batch_start_id + i
                    chunk_data = self.generate_chunk_data_with_timestamp(record_id, batch_base_time)
                    batch_data.append({
                        "id": record_id,
                        "chunk_data": chunk_data
                    })

                # Submit batch for processing
                worker_id = batch_idx % max_workers
                future = executor.submit(self.bulk_insert_optimized, batch_data, worker_id)
                futures[future] = current_batch_size

                # Process completed futures to control memory usage
                if len(futures) >= max_workers * 3:  # Keep some futures in flight
                    completed = []
                    for future in as_completed(futures, timeout=30):
                        result = future.result()
                        batch_size = futures[future]
                        
                        if result > 0:
                            successful_batches += 1
                            records_inserted += result
                        else:
                            failed_batches += 1
                            
                        pbar.update(batch_size)
                        completed.append(future)
                        break  # Process one at a time to avoid blocking
                    
                    # Remove completed futures
                    for future in completed:
                        del futures[future]

            # Wait for remaining futures
            for future in as_completed(futures):
                result = future.result()
                batch_size = futures[future]
                
                if result > 0:
                    successful_batches += 1
                    records_inserted += result
                else:
                    failed_batches += 1
                    
                pbar.update(batch_size)

        pbar.close()

        # Final statistics
        end_time = time.time()
        duration = end_time - start_population_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Get final count and TimescaleDB stats
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                final_count = cur.fetchone()[0]
        except Exception as e:
            final_count = current_count + records_inserted
        finally:
            conn.close()

        print(f"\n{'='*70}")
        print(f"TIMESCALEDB POPULATION COMPLETED")
        print(f"{'='*70}")
        print(f"Records added: {records_inserted:,}")
        print(f"Previous count: {current_count:,}")
        print(f"Final count: {final_count:,}")
        print(f"Successful batches: {successful_batches:,}")
        print(f"Failed batches: {failed_batches:,}")
        print(f"Duration: {duration:.2f} seconds")
        if duration > 0:
            print(f"Records per second: {records_inserted / duration:,.0f}")
        print(f"Memory used: {final_memory - initial_memory:.2f} MB")
        print(f"Peak memory: {final_memory:.2f} MB")

    def get_collection_stats(self):
        """Get TimescaleDB-specific collection statistics"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Try schema's stats function first
                try:
                    cur.execute("SELECT * FROM get_collection_stats_detailed();")
                    stats = cur.fetchone()
                    
                    print(f"\n{'='*70}")
                    print(f"OPTIMIZED TIMESCALEDB STATISTICS")
                    print(f"{'='*70}")
                    print(f"Total vectors: {stats['total_vectors']:,}")
                    print(f"Total chunks: {stats['total_chunks']}")
                    print(f"Table size: {stats['table_size']}")
                    print(f"Vector dimensions: {stats['vector_dimensions']}")
                    
                except Exception:
                    # Fallback to basic TimescaleDB stats
                    cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                    total_count = cur.fetchone()[0]
                    
                    cur.execute("""
                        SELECT COUNT(*) as chunk_count, 
                               SUM(CASE WHEN is_compressed THEN 1 ELSE 0 END) as compressed_chunks
                        FROM timescaledb_information.chunks
                        WHERE hypertable_schema = %s AND hypertable_name = 'embeddings_ts';
                    """, (self.schema,))
                    chunk_stats = cur.fetchone()
                    
                    cur.execute(f"SELECT pg_size_pretty(hypertable_size('{self.table_name}'));")
                    table_size = cur.fetchone()[0]
                    
                    print(f"\n{'='*70}")
                    print(f"TIMESCALEDB STATISTICS")
                    print(f"{'='*70}")
                    print(f"Total vectors: {total_count:,}")
                    print(f"Vector dimensions: {self.vector_dim}")
                    print(f"Total chunks: {chunk_stats[0] if chunk_stats else 0}")
                    print(f"Compressed chunks: {chunk_stats[1] if chunk_stats else 0}")
                    print(f"Hypertable size: {table_size}")
                    print(f"Index type: DiskANN (pgvectorscale)")

        except Exception as e:
            print(f"Error getting collection stats: {e}")
        finally:
            conn.close()

    def test_vector_search_with_time_filtering(self, num_queries: int = 5):
        """Test TimescaleDB-optimized vector search with time filtering"""
        print(f"\nTesting TimescaleDB vector search with time filtering ({num_queries} queries)...")
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Apply vectorscale optimizations
                cur.execute("SET vectorscale.query_rescore = 200;")
                cur.execute("SET work_mem = '512MB';")
                
                # Get time range
                cur.execute(f"SELECT MIN(created_at), MAX(created_at) FROM {self.table_name};")
                time_range = cur.fetchone()
                
                if time_range[0] is None:
                    print("No data found for search testing")
                    return

                min_time, max_time = time_range
                time_span = (max_time - min_time).total_seconds()
                
                total_time_all = 0
                total_time_filtered = 0

                for i in range(num_queries):
                    query_vector = self.generate_random_vector()
                    
                    # Test 1: All-time search
                    start_time = time.time()
                    try:
                        # Try optimized function
                        cur.execute("""
                            SELECT * FROM search_similar_vectors(%s, 10, 0.0);
                        """, (query_vector,))
                    except Exception:
                        # Fallback to direct query
                        cur.execute(f"""
                            SELECT vector_id, text_content, metadata,
                                   1 - (embedding <=> %s::vector) as similarity
                            FROM {self.table_name}
                            ORDER BY embedding <=> %s::vector
                            LIMIT 10;
                        """, (query_vector, query_vector))
                    
                    results_all = cur.fetchall()
                    search_time_all = time.time() - start_time
                    total_time_all += search_time_all

                    # Test 2: Time-filtered search (last 20% of data)
                    filter_start = min_time + timedelta(seconds=time_span * 0.8)
                    
                    start_time = time.time()
                    try:
                        # Try time-range function
                        cur.execute("""
                            SELECT * FROM search_similar_vectors_time_range(%s, %s, %s, 10);
                        """, (query_vector, filter_start, max_time))
                    except Exception:
                        # Fallback
                        cur.execute(f"""
                            SELECT vector_id, text_content, metadata, created_at,
                                   1 - (embedding <=> %s::vector) as similarity
                            FROM {self.table_name}
                            WHERE created_at >= %s
                            ORDER BY embedding <=> %s::vector
                            LIMIT 10;
                        """, (query_vector, filter_start, query_vector))
                    
                    results_filtered = cur.fetchall()
                    search_time_filtered = time.time() - start_time
                    total_time_filtered += search_time_filtered

                    if i == 0:
                        print(f"Sample results:")
                        print(f"  All-time: {len(results_all)} matches, top similarity: {results_all[0][3]:.4f}")
                        print(f"  Time-filtered: {len(results_filtered)} matches, top similarity: {results_filtered[0][-1]:.4f}")

                avg_time_all = total_time_all / num_queries
                avg_time_filtered = total_time_filtered / num_queries
                
                print(f"\nTimescaleDB Search Performance:")
                print(f"All-time search: {avg_time_all:.4f}s ({1/avg_time_all:.2f} QPS)")
                print(f"Time-filtered search: {avg_time_filtered:.4f}s ({1/avg_time_filtered:.2f} QPS)")
                
                if avg_time_all > avg_time_filtered:
                    speedup = ((avg_time_all - avg_time_filtered) / avg_time_all) * 100
                    print(f"Time filtering speedup: {speedup:.1f}%")

        except Exception as e:
            print(f"Error during search test: {e}")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Populate optimized TimescaleDB vector database")
    parser.add_argument("--host", default="localhost", help="TimescaleDB host")
    parser.add_argument("--port", type=int, default=5433, help="TimescaleDB port")
    parser.add_argument("--database", default="vectordb", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--schema", default="public", help="Database schema")
    parser.add_argument("--records", type=int, default=10_000_000, help="Number of records to insert")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for inserts")
    parser.add_argument("--time-spread-days", type=int, default=30, help="Spread data across N days")
    parser.add_argument("--show-stats", action="store_true", help="Show collection stats")
    parser.add_argument("--test-search", action="store_true", help="Test search performance")

    args = parser.parse_args()

    populator = OptimizedTimescaleDBVectorPopulator(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        schema=args.schema
    )
    populator.batch_size = args.batch_size

    try:
        populator.populate_database(
            total_records=args.records,
            max_workers=args.workers,
            time_spread_days=args.time_spread_days
        )

        if args.show_stats:
            populator.get_collection_stats()
            
        if args.test_search:
            populator.test_vector_search_with_time_filtering()

    except KeyboardInterrupt:
        print("\nPopulation interrupted by user")
    except Exception as e:
        print(f"Error during population: {e}")
        raise


if __name__ == "__main__":
    main()
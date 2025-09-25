#!/usr/bin/env python3
"""
TimescaleDB Vector Database Population Script
Populates TimescaleDB with pgvectorscale extension with test data.
Optimized for hypertables, compression, and StreamingDiskANN indexing.
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
from datetime import datetime, timedelta


class TimescaleDBVectorPopulator:
    def __init__(self, host: str = "localhost", port: int = 5433,
                 database: str = "vectordb", user: str = "postgres",
                 password: str = "postgres"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.vector_dim = 1024
        self.batch_size = 500  # Smaller batches to reduce lock contention
        self.chunk_time_interval = "1 day"  # TimescaleDB hypertable chunk interval

    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )

    def check_extensions(self):
        """Check and install required extensions"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Check for TimescaleDB extension
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
                timescaledb_exists = cur.fetchone() is not None

                if not timescaledb_exists:
                    print("Installing TimescaleDB extension...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                else:
                    print("OK: TimescaleDB extension already installed")

                # Check for pgvector extension
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                pgvector_exists = cur.fetchone() is not None

                if not pgvector_exists:
                    print("Installing pgvector extension...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector CASCADE;")
                else:
                    print("OK: pgvector extension already installed")

                # Check for pgvectorscale extension
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vectorscale';")
                pgvectorscale_exists = cur.fetchone() is not None

                if not pgvectorscale_exists:
                    print("Installing pgvectorscale extension...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
                else:
                    print("OK: pgvectorscale extension already installed")

                conn.commit()

        except Exception as e:
            print(f"Error checking/installing extensions: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def create_hypertable(self):
        """Create the vector embeddings hypertable if it doesn't exist"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'vector_embeddings_ts'
                    );
                """)
                table_exists = cur.fetchone()[0]

                if table_exists:
                    print("Table 'vector_embeddings_ts' already exists. Adding to existing table...")

                    # Check if it's already a hypertable
                    cur.execute("""
                        SELECT * FROM timescaledb_information.hypertables
                        WHERE hypertable_name = 'vector_embeddings_ts';
                    """)
                    is_hypertable = cur.fetchone() is not None

                    if not is_hypertable:
                        print("WARNING: Table exists but is not a hypertable. Consider recreating for optimal performance.")
                    else:
                        print("OK: Table is already a hypertable - optimized for time-series data")

                    # Check vector dimension compatibility
                    cur.execute("""
                        SELECT column_name, data_type, udt_name, character_maximum_length
                        FROM information_schema.columns
                        WHERE table_name = 'vector_embeddings_ts'
                        AND column_name = 'embedding';
                    """)
                    result = cur.fetchone()

                    if result:
                        data_type = result[1]
                        if 'vector' in data_type.lower():
                            import re
                            match = re.search(r'VECTOR\((\d+)\)', data_type)
                            if match:
                                current_dim = int(match.group(1))
                                if current_dim != self.vector_dim:
                                    print(f"WARNING: Table has {current_dim}D vectors, but script is using {self.vector_dim}D vectors.")
                                    print("WARNING: This may cause errors. Consider using --vector-dim {current_dim} to match existing data.")
                                else:
                                    print(f"OK: Table ready for {self.vector_dim}D vectors - will add to existing data")
                else:
                    print("Table 'vector_embeddings_ts' does not exist. Creating hypertable...")
                    self._create_hypertable_with_dimensions(cur, self.vector_dim)
                    print(f"OK: Hypertable created with {self.vector_dim}D vectors and {self.chunk_time_interval} chunks")

                conn.commit()

        except Exception as e:
            print(f"Error creating hypertable: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_hypertable_with_dimensions(self, cur, vector_dim):
        """Create hypertable with specified vector dimensions and TimescaleDB optimizations"""
        # Create table
        cur.execute(f"""
            CREATE TABLE vector_embeddings_ts (
                id SERIAL,
                vector_id INTEGER NOT NULL,
                embedding VECTOR({vector_dim}),
                text_content TEXT,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Convert to hypertable with time partitioning
        cur.execute(f"""
            SELECT create_hypertable('vector_embeddings_ts', 'created_at',
                                     chunk_time_interval => INTERVAL '{self.chunk_time_interval}');
        """)

        # Create indexes optimized for TimescaleDB
        cur.execute("CREATE INDEX idx_vector_embeddings_ts_vector_id ON vector_embeddings_ts(vector_id, created_at DESC);")
        cur.execute("CREATE INDEX idx_vector_embeddings_ts_metadata ON vector_embeddings_ts USING GIN(metadata);")

        # Create StreamingDiskANN index (pgvectorscale optimization)
        cur.execute(f"""
            CREATE INDEX idx_vector_embeddings_ts_embedding_diskann
            ON vector_embeddings_ts USING diskann (embedding vector_cosine_ops);
        """)

        # Enable compression (TimescaleDB feature for older chunks)
        cur.execute("""
            ALTER TABLE vector_embeddings_ts SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'vector_id',
                timescaledb.compress_orderby = 'created_at DESC'
            );
        """)

        # Create compression policy (compress chunks older than 7 days)
        cur.execute("""
            SELECT add_compression_policy('vector_embeddings_ts', INTERVAL '7 days');
        """)

        # Create retention policy (optional - keep data for 1 year)
        cur.execute("""
            SELECT add_retention_policy('vector_embeddings_ts', INTERVAL '1 year');
        """)

        # Create trigger function for updated_at
        cur.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)

        # Create trigger
        cur.execute("""
            CREATE TRIGGER update_vector_embeddings_ts_updated_at
                BEFORE UPDATE ON vector_embeddings_ts
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)

        # Create optimized search function using pgvectorscale
        cur.execute(f"""
            CREATE OR REPLACE FUNCTION search_similar_vectors_ts(
                query_embedding VECTOR({vector_dim}),
                match_limit INTEGER DEFAULT 10,
                similarity_threshold FLOAT DEFAULT 0.0,
                time_start TIMESTAMPTZ DEFAULT NULL,
                time_end TIMESTAMPTZ DEFAULT NULL
            )
            RETURNS TABLE (
                vector_id INTEGER,
                text_content TEXT,
                metadata JSONB,
                similarity FLOAT,
                created_at TIMESTAMPTZ
            ) AS $$
            BEGIN
                -- Set pgvectorscale parameters for optimal performance
                SET LOCAL diskann.query_search_list_size = 100;
                SET LOCAL diskann.query_rescore = 200;

                RETURN QUERY
                SELECT
                    ve.vector_id,
                    ve.text_content,
                    ve.metadata,
                    1 - (ve.embedding <=> query_embedding) AS similarity,
                    ve.created_at
                FROM vector_embeddings_ts ve
                WHERE 1 - (ve.embedding <=> query_embedding) > similarity_threshold
                    AND (time_start IS NULL OR ve.created_at >= time_start)
                    AND (time_end IS NULL OR ve.created_at <= time_end)
                ORDER BY ve.embedding <=> query_embedding
                LIMIT match_limit;
            END;
            $$ LANGUAGE plpgsql;
        """)

        # Create stats function
        cur.execute(f"""
            CREATE OR REPLACE FUNCTION get_timescale_collection_stats()
            RETURNS TABLE (
                total_points BIGINT,
                vector_dimensions INTEGER,
                avg_metadata_size FLOAT,
                created_at_range TEXT,
                hypertable_size TEXT,
                chunk_count BIGINT,
                compressed_chunks BIGINT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT
                    COUNT(*)::BIGINT as total_points,
                    {vector_dim} as vector_dimensions,
                    AVG(LENGTH(metadata::TEXT)::FLOAT) as avg_metadata_size,
                    CONCAT(
                        MIN(created_at)::TEXT,
                        ' to ',
                        MAX(created_at)::TEXT
                    ) as created_at_range,
                    pg_size_pretty(hypertable_size('vector_embeddings_ts')) as hypertable_size,
                    (SELECT COUNT(*) FROM timescaledb_information.chunks
                     WHERE hypertable_name = 'vector_embeddings_ts') as chunk_count,
                    (SELECT COUNT(*) FROM timescaledb_information.chunks
                     WHERE hypertable_name = 'vector_embeddings_ts'
                     AND is_compressed = true) as compressed_chunks
                FROM vector_embeddings_ts;
            END;
            $$ LANGUAGE plpgsql;
        """)

    def generate_random_vector(self) -> List[float]:
        """Generate a random normalized vector"""
        vector = np.random.normal(0, 1, self.vector_dim)
        # Normalize the vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def generate_chunk_data_with_timestamp(self, chunk_id: int, base_time: datetime) -> Dict[str, Any]:
        """Generate sample chunk data with realistic timestamps for time-series data"""
        # Add some randomness to timestamps for more realistic time-series data
        time_offset = timedelta(
            minutes=random.randint(0, 1440),  # Random within 24 hours
            seconds=random.randint(0, 3600)   # Random seconds within an hour
        )
        record_time = base_time + time_offset

        return {
            "id": chunk_id,
            "text": f"This is time-series chunk number {chunk_id} created at {record_time.isoformat()}. " * random.randint(1, 3),
            "metadata": {
                "source": f"timeseries_document_{chunk_id % 1000}",
                "page": random.randint(1, 100),
                "section": random.choice(["introduction", "body", "conclusion", "appendix"]),
                "timestamp": int(record_time.timestamp()),
                "category": random.choice(["technical", "business", "legal", "scientific", "medical"]),
                "confidence": round(random.uniform(0.5, 1.0), 3),
                "sensor_id": f"sensor_{random.randint(1, 100)}",
                "location": random.choice(["datacenter_a", "datacenter_b", "edge_location"]),
                "priority": random.choice(["low", "medium", "high", "critical"])
            },
            "created_at": record_time
        }

    def get_next_available_id(self) -> int:
        """Get the next available ID to avoid conflicts when adding to existing data"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) + 1 FROM vector_embeddings_ts;")
                next_id = cur.fetchone()[0]
                return next_id
        except Exception as e:
            print(f"Error getting next available ID: {e}")
            return 1  # Fallback to starting from 1
        finally:
            conn.close()

    def insert_batch_with_timestamps(self, batch_data: List[Dict[str, Any]]) -> bool:
        """Insert a batch of records with time-series optimization"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Set TimescaleDB/pgvectorscale optimization parameters
                cur.execute("SET work_mem = '256MB';")
                cur.execute("SET maintenance_work_mem = '1GB';")

                # Reduce lock timeout to fail fast on deadlocks
                cur.execute("SET lock_timeout = '5s';")
                cur.execute("SET deadlock_timeout = '1s';")

                # Use a single transaction with immediate commit
                # Prepare batch insert (no ON CONFLICT since we don't have unique constraint)
                insert_query = """
                    INSERT INTO vector_embeddings_ts (vector_id, embedding, text_content, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """

                # Convert data to tuples for batch insert
                batch_tuples = []
                for record in batch_data:
                    vector = self.generate_random_vector()
                    chunk_data = record['chunk_data']

                    batch_tuples.append((
                        record['id'],
                        vector,
                        chunk_data['text'],
                        json.dumps(chunk_data['metadata']),
                        chunk_data['created_at']
                    ))

                # Execute batch insert
                cur.executemany(insert_query, batch_tuples)
                conn.commit()
                return True

        except psycopg2.errors.DeadlockDetected as e:
            # Retry once on deadlock
            conn.rollback()
            try:
                import time
                import random
                time.sleep(random.uniform(0.1, 0.5))  # Random backoff
                with conn.cursor() as cur:
                    cur.executemany(insert_query, batch_tuples)
                    conn.commit()
                    return True
            except Exception:
                print(f"Deadlock retry failed for batch")
                return False
        except Exception as e:
            print(f"Error inserting batch: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def populate_database(self, total_records: int = 10_000_000, max_workers: int = 2,
                         time_spread_days: int = 30):
        """Populate the database with time-series optimized records"""
        print(f"\n{'='*60}")
        print(f"POPULATING TIMESCALEDB DATABASE")
        print(f"{'='*60}")
        print(f"Database: {self.database}")
        print(f"Records to insert: {total_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max workers: {max_workers}")
        print(f"Time spread: {time_spread_days} days")
        print(f"Chunk interval: {self.chunk_time_interval}")

        # Get current count and next available ID
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings_ts;")
                current_count = cur.fetchone()[0]
                print(f"Current records in hypertable: {current_count:,}")
        except Exception as e:
            print(f"Error getting current count: {e}")
            current_count = 0
        finally:
            conn.close()

        # Get the next available ID to avoid conflicts
        start_id = self.get_next_available_id()
        print(f"Starting from ID: {start_id}")
        print(f"Will add {total_records:,} new records (total will be {current_count + total_records:,})")

        # Create time range for realistic time-series data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_spread_days)
        print(f"Time range: {start_time.isoformat()} to {end_time.isoformat()}")

        start_population_time = time.time()
        total_batches = (total_records + self.batch_size - 1) // self.batch_size

        # Create progress bar
        pbar = tqdm(total=total_records, desc="Inserting time-series records", unit="records")

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

                # Create batch data with time-series distribution
                batch_data = []
                for i in range(current_batch_size):
                    record_id = batch_start_id + i
                    # Distribute records across time range
                    time_progress = (batch_idx * self.batch_size + i) / total_records
                    record_time = start_time + timedelta(
                        seconds=int(time_progress * time_spread_days * 24 * 3600)
                    )

                    chunk_data = self.generate_chunk_data_with_timestamp(record_id, record_time)
                    batch_data.append({
                        "id": record_id,
                        "chunk_data": chunk_data
                    })

                # Submit batch for insertion
                future = executor.submit(self.insert_batch_with_timestamps, batch_data)
                futures.append(future)

                # Process completed futures to manage memory
                if len(futures) >= max_workers * 2:  # Keep some futures in flight
                    completed_futures = []
                    for future in futures:
                        if future.done():
                            if future.result():
                                successful_batches += 1
                            else:
                                failed_batches += 1
                            completed_futures.append(future)
                            pbar.update(self.batch_size)

                    # Remove completed futures
                    for future in completed_futures:
                        futures.remove(future)

            # Wait for remaining futures
            for future in futures:
                if future.result():
                    successful_batches += 1
                else:
                    failed_batches += 1
                pbar.update(self.batch_size)

        pbar.close()

        end_population_time = time.time()
        duration = end_population_time - start_population_time

        # Final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        # Get final count
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings_ts;")
                final_count = cur.fetchone()[0]
        except Exception as e:
            print(f"Error getting final count: {e}")
            final_count = current_count + total_records
        finally:
            conn.close()

        # Print statistics
        print(f"\n{'='*60}")
        print(f"TIMESCALEDB POPULATION COMPLETED")
        print(f"{'='*60}")
        print(f"Records added: {total_records:,}")
        print(f"Previous count: {current_count:,}")
        print(f"Final count: {final_count:,}")
        print(f"Successful batches: {successful_batches:,}")
        print(f"Failed batches: {failed_batches:,}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Records per second: {total_records / duration:,.0f}")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory: {final_memory:.2f} MB")

        # Get TimescaleDB-specific statistics
        self.get_collection_stats()

    def get_collection_stats(self):
        """Get and display TimescaleDB-specific collection statistics"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Use our custom stats function
                cur.execute("SELECT * FROM get_timescale_collection_stats();")
                stats = cur.fetchone()

                # Get chunk information (simplified to avoid column name issues)
                cur.execute("""
                    SELECT
                        chunk_name,
                        range_start,
                        range_end,
                        is_compressed
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = 'vector_embeddings_ts'
                    ORDER BY range_start DESC
                    LIMIT 5;
                """)
                recent_chunks = cur.fetchall()

                print(f"\n{'='*60}")
                print(f"TIMESCALEDB COLLECTION STATISTICS")
                print(f"{'='*60}")
                print(f"Database: {self.database}")
                print(f"Total points: {stats['total_points']:,}")
                print(f"Vector dimensions: {stats['vector_dimensions']}")
                print(f"Average metadata size: {stats['avg_metadata_size']:.1f} characters")
                print(f"Time range: {stats['created_at_range']}")
                print(f"Hypertable size: {stats['hypertable_size']}")
                print(f"Total chunks: {stats['chunk_count']}")
                print(f"Compressed chunks: {stats['compressed_chunks']}")
                print(f"Index type: StreamingDiskANN (pgvectorscale)")
                print(f"Distance metric: COSINE")
                print(f"Chunk interval: {self.chunk_time_interval}")

                print(f"\nRecent chunks:")
                for chunk in recent_chunks:
                    compressed = "Compressed" if chunk['is_compressed'] else "Uncompressed"
                    print(f"  {chunk['chunk_name']}: {chunk['range_start']} to {chunk['range_end']} - {compressed}")

        except Exception as e:
            print(f"Error getting collection stats: {e}")
        finally:
            conn.close()

    def test_search_with_time_filter(self, num_queries: int = 10):
        """Test vector similarity search with time-based filtering (TimescaleDB optimization)"""
        print(f"\nTesting TimescaleDB vector search with time filtering ({num_queries} queries)...")

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Set pgvectorscale parameters for optimal performance
                cur.execute("SET diskann.query_search_list_size = 100;")
                cur.execute("SET diskann.query_rescore = 200;")

                # Get time range for filtering
                cur.execute("SELECT MIN(created_at), MAX(created_at) FROM vector_embeddings_ts;")
                time_range = cur.fetchone()
                if time_range[0] is None:
                    print("No data found for search testing")
                    return

                min_time, max_time = time_range
                time_span = (max_time - min_time).total_seconds()

                total_time = 0
                total_time_filtered = 0

                for i in range(num_queries):
                    # Generate random query vector
                    query_vector = self.generate_random_vector()

                    # Test 1: Search without time filter
                    start_time = time.time()
                    cur.execute("""
                        SELECT vector_id, text_content, metadata,
                               1 - (embedding <=> %s::vector) as similarity,
                               created_at
                        FROM vector_embeddings_ts
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, query_vector))

                    results_all = cur.fetchall()
                    search_time = time.time() - start_time
                    total_time += search_time

                    # Test 2: Search with time filter (last 25% of time range)
                    filter_start = min_time + timedelta(seconds=time_span * 0.75)

                    start_time_filtered = time.time()
                    cur.execute("""
                        SELECT vector_id, text_content, metadata,
                               1 - (embedding <=> %s::vector) as similarity,
                               created_at
                        FROM vector_embeddings_ts
                        WHERE created_at >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, filter_start, query_vector))

                    results_filtered = cur.fetchall()
                    search_time_filtered = time.time() - start_time_filtered
                    total_time_filtered += search_time_filtered

                    if i == 0:  # Show first result as example
                        print(f"Sample search results:")
                        print(f"  All-time search: {len(results_all)} matches, top similarity: {results_all[0][3]:.4f}")
                        print(f"  Time-filtered search: {len(results_filtered)} matches, top similarity: {results_filtered[0][3]:.4f}")

                avg_search_time = total_time / num_queries
                avg_search_time_filtered = total_time_filtered / num_queries

                print(f"\nSearch Performance:")
                print(f"Average search time (all-time): {avg_search_time:.4f}s")
                print(f"Average search time (time-filtered): {avg_search_time_filtered:.4f}s")
                print(f"All-time search QPS: {1/avg_search_time:.2f}")
                print(f"Time-filtered search QPS: {1/avg_search_time_filtered:.2f}")
                print(f"Time-filtering performance gain: {((avg_search_time - avg_search_time_filtered) / avg_search_time * 100):.1f}%")

        except Exception as e:
            print(f"Error during search test: {e}")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Populate TimescaleDB vector database with time-series test data")
    parser.add_argument("--host", default="localhost", help="TimescaleDB host")
    parser.add_argument("--port", type=int, default=5433, help="TimescaleDB port")
    parser.add_argument("--database", default="vectordb", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--records", type=int, default=10_000_000, help="Number of records to insert")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    parser.add_argument("--time-spread-days", type=int, default=30, help="Spread data across N days")
    parser.add_argument("--chunk-interval", default="1 day", help="TimescaleDB chunk time interval")
    parser.add_argument("--test-search", action="store_true", help="Run search test after population")

    args = parser.parse_args()

    # Create populator instance
    populator = TimescaleDBVectorPopulator(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )
    populator.vector_dim = args.vector_dim
    populator.chunk_time_interval = args.chunk_interval

    try:
        # Check and install required extensions
        populator.check_extensions()

        # Create hypertable
        populator.create_hypertable()

        # Populate database
        populator.populate_database(
            total_records=args.records,
            max_workers=args.workers,
            time_spread_days=args.time_spread_days
        )

        # Test search if requested
        if args.test_search:
            populator.test_search_with_time_filter()

    except KeyboardInterrupt:
        print("\nPopulation interrupted by user")
    except Exception as e:
        print(f"Error during population: {e}")
        raise


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Optimized test data generation for PostgreSQL + pgvector and TimescaleDB + vectorscale
Designed for benchmarking with 100K, 500K, 1M, and 5M records
"""

import psycopg2
import numpy as np
import time
import argparse
import multiprocessing as mp
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import json
from typing import List, Dict, Any
import random
import string

class VectorDatabasePopulator:
    def __init__(self, postgres_config: Dict, timescaledb_config: Dict):
        self.postgres_config = postgres_config
        self.timescaledb_config = timescaledb_config
        self.vector_dim = 1024

    def generate_batch_data(self, batch_size: int, start_id: int, time_spread_days: int = 30) -> List[tuple]:
        """Generate a batch of test data optimized for vector databases"""
        batch_data = []
        base_time = datetime.now() - timedelta(days=time_spread_days)

        for i in range(batch_size):
            vector_id = start_id + i

            # Generate normalized random vector (important for vector similarity)
            embedding = np.random.normal(0, 1, self.vector_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize

            # Generate realistic metadata
            text_content = f"Sample document {vector_id} with content " + ''.join(random.choices(string.ascii_lowercase, k=50))

            metadata = {
                "category": random.choice(["tech", "science", "business", "health", "education"]),
                "priority": random.randint(1, 10),
                "tags": random.sample(["ai", "ml", "data", "cloud", "security", "mobile", "web"], k=random.randint(1, 4)),
                "source": random.choice(["web", "api", "upload", "import"])
            }

            # Spread timestamps across the time period (important for partitioning)
            random_offset = random.randint(0, time_spread_days * 24 * 60 * 60)
            created_at = base_time + timedelta(seconds=random_offset)

            batch_data.append((
                vector_id,
                embedding.tolist(),
                text_content,
                json.dumps(metadata),
                created_at
            ))

        return batch_data

    def insert_batch_postgres(self, batch_data: List[tuple], connection_config: Dict) -> Dict[str, Any]:
        """Optimized batch insertion for PostgreSQL with partitioning"""
        start_time = time.time()

        try:
            # Use separate connection for each thread
            conn = psycopg2.connect(**connection_config)
            conn.autocommit = False
            cur = conn.cursor()

            # Use execute_values for better compatibility with vector arrays
            from psycopg2.extras import execute_values

            insert_sql = """
                INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata, created_at)
                VALUES %s
            """

            # Format data for execute_values
            formatted_data = []
            for vector_id, embedding, text_content, metadata, created_at in batch_data:
                formatted_data.append((
                    vector_id,
                    embedding,  # psycopg2 handles list conversion automatically
                    text_content,
                    metadata,
                    created_at
                ))

            # Use execute_values with optimal page size
            execute_values(
                cur,
                insert_sql,
                formatted_data,
                page_size=1000,
                fetch=False
            )
            conn.commit()

            duration = time.time() - start_time

            cur.close()
            conn.close()

            return {
                "success": True,
                "batch_size": len(batch_data),
                "duration": duration,
                "records_per_second": len(batch_data) / duration
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "batch_size": len(batch_data),
                "duration": time.time() - start_time
            }

    def insert_batch_timescaledb(self, batch_data: List[tuple], connection_config: Dict) -> Dict[str, Any]:
        """Optimized batch insertion for TimescaleDB hypertable"""
        start_time = time.time()

        try:
            # Use separate connection for each thread
            conn = psycopg2.connect(**connection_config)
            conn.autocommit = False
            cur = conn.cursor()

            # Use execute_values for TimescaleDB (optimized for hypertables)
            from psycopg2.extras import execute_values

            insert_sql = """
                INSERT INTO vector_embeddings_ts (vector_id, embedding, text_content, metadata, created_at)
                VALUES %s
            """

            # Convert embedding lists to strings for psycopg2
            formatted_data = []
            for vector_id, embedding, text_content, metadata, created_at in batch_data:
                formatted_data.append((
                    vector_id,
                    embedding,  # psycopg2 handles list conversion automatically
                    text_content,
                    metadata,
                    created_at
                ))

            # Use execute_values with optimal page size for TimescaleDB
            execute_values(
                cur,
                insert_sql,
                formatted_data,
                page_size=1000,  # Optimal for TimescaleDB chunks
                fetch=False
            )

            conn.commit()
            duration = time.time() - start_time

            cur.close()
            conn.close()

            return {
                "success": True,
                "batch_size": len(batch_data),
                "duration": duration,
                "records_per_second": len(batch_data) / duration
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "batch_size": len(batch_data),
                "duration": time.time() - start_time
            }

    def populate_database(self, total_records: int, database_type: str, batch_size: int = 10000, max_workers: int = None) -> Dict[str, Any]:
        """Populate database with optimal strategy for each type"""

        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Limit concurrent connections

        config = self.postgres_config if database_type == "postgres" else self.timescaledb_config
        insert_func = self.insert_batch_postgres if database_type == "postgres" else self.insert_batch_timescaledb
        table_name = "vector_embeddings" if database_type == "postgres" else "vector_embeddings_ts"

        print(f"\n🚀 Populating {database_type.upper()} database")
        print(f"📊 Target records: {total_records:,}")
        print(f"📦 Batch size: {batch_size:,}")
        print(f"⚡ Workers: {max_workers}")
        print(f"📋 Table: {table_name}")

        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Calculate batches
        num_batches = (total_records + batch_size - 1) // batch_size
        successful_batches = 0
        failed_batches = 0
        total_records_inserted = 0

        # Generate and process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch jobs
            futures = []
            for batch_idx in range(num_batches):
                start_id = batch_idx * batch_size + 1
                actual_batch_size = min(batch_size, total_records - batch_idx * batch_size)

                # Generate batch data
                batch_data = self.generate_batch_data(actual_batch_size, start_id)

                # Submit insertion job
                future = executor.submit(insert_func, batch_data, config)
                futures.append((future, batch_idx, actual_batch_size))

            # Process completed jobs with progress tracking
            for i, (future, batch_idx, actual_batch_size) in enumerate(futures):
                try:
                    result = future.result()

                    if result["success"]:
                        successful_batches += 1
                        total_records_inserted += result["batch_size"]

                        # Progress indicator
                        progress = (i + 1) / len(futures) * 100
                        records_per_sec = result["records_per_second"]
                        print(f"✅ Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%) - {records_per_sec:.0f} records/sec")
                    else:
                        failed_batches += 1
                        print(f"❌ Batch {batch_idx + 1} failed: {result.get('error', 'Unknown error')}")

                except Exception as e:
                    failed_batches += 1
                    print(f"❌ Batch {batch_idx + 1} exception: {e}")

        total_duration = time.time() - start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Calculate final statistics
        overall_records_per_sec = total_records_inserted / total_duration if total_duration > 0 else 0

        # Verify actual record count in database
        try:
            conn = psycopg2.connect(**config)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            actual_count = cur.fetchone()[0]
            cur.close()
            conn.close()
        except Exception as e:
            actual_count = "Unknown"
            print(f"⚠️  Could not verify record count: {e}")

        results = {
            "database_type": database_type,
            "target_records": total_records,
            "records_inserted": total_records_inserted,
            "actual_db_count": actual_count,
            "successful_batches": successful_batches,
            "failed_batches": failed_batches,
            "total_duration": total_duration,
            "records_per_second": overall_records_per_sec,
            "memory_used_mb": final_memory - initial_memory,
            "peak_memory_mb": final_memory,
            "batch_size": batch_size,
            "workers": max_workers
        }

        # Print summary
        print(f"\n📊 {database_type.upper()} POPULATION SUMMARY")
        print("=" * 50)
        print(f"Target records: {total_records:,}")
        print(f"Records inserted: {total_records_inserted:,}")
        print(f"Actual DB count: {actual_count:,}" if isinstance(actual_count, int) else f"Actual DB count: {actual_count}")
        print(f"Successful batches: {successful_batches}/{num_batches}")
        print(f"Failed batches: {failed_batches}")
        print(f"Duration: {total_duration:.2f} seconds")
        print(f"Records per second: {overall_records_per_sec:.0f}")
        print(f"Memory used: {final_memory - initial_memory:.1f} MB")
        print(f"Peak memory: {final_memory:.1f} MB")

        return results

def main():
    parser = argparse.ArgumentParser(description="Populate PostgreSQL and TimescaleDB with vector test data")
    parser.add_argument("--records", type=int, default=100000, help="Number of records to generate (default: 100,000)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for insertions (default: 10,000)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: auto)")
    parser.add_argument("--database", choices=["postgres", "timescaledb", "both"], default="both", help="Which database to populate")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--timescaledb-host", default="localhost", help="TimescaleDB host")
    parser.add_argument("--timescaledb-port", type=int, default=5433, help="TimescaleDB port")
    parser.add_argument("--username", default="postgres", help="Database username")
    parser.add_argument("--password", default="password", help="Database password")
    parser.add_argument("--database-name", default="vectordb", help="Database name")

    args = parser.parse_args()

    # Database configurations
    postgres_config = {
        "host": args.postgres_host,
        "port": args.postgres_port,
        "user": args.username,
        "password": args.password,
        "database": args.database_name
    }

    timescaledb_config = {
        "host": args.timescaledb_host,
        "port": args.timescaledb_port,
        "user": args.username,
        "password": args.password,
        "database": args.database_name
    }

    # Create populator
    populator = VectorDatabasePopulator(postgres_config, timescaledb_config)

    results = {}

    print(f"🎯 VECTOR DATABASE POPULATION BENCHMARK")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔢 Records: {args.records:,}")
    print(f"📦 Batch size: {args.batch_size:,}")
    print(f"⚡ Workers: {args.workers or 'auto'}")

    # Populate databases
    if args.database in ["postgres", "both"]:
        try:
            results["postgres"] = populator.populate_database(
                args.records, "postgres", args.batch_size, args.workers
            )
        except Exception as e:
            print(f"❌ PostgreSQL population failed: {e}")
            results["postgres"] = {"error": str(e)}

    if args.database in ["timescaledb", "both"]:
        try:
            results["timescaledb"] = populator.populate_database(
                args.records, "timescaledb", args.batch_size, args.workers
            )
        except Exception as e:
            print(f"❌ TimescaleDB population failed: {e}")
            results["timescaledb"] = {"error": str(e)}

    # Comparative summary if both databases were populated
    if len(results) == 2 and "error" not in results.get("postgres", {}) and "error" not in results.get("timescaledb", {}):
        print(f"\n🏆 COMPARATIVE RESULTS")
        print("=" * 50)

        pg_results = results["postgres"]
        ts_results = results["timescaledb"]

        print(f"PostgreSQL:")
        print(f"  📊 {pg_results['records_per_second']:.0f} records/sec")
        print(f"  ⏱️  {pg_results['total_duration']:.2f} seconds")
        print(f"  💾 {pg_results['memory_used_mb']:.1f} MB memory")

        print(f"TimescaleDB:")
        print(f"  📊 {ts_results['records_per_second']:.0f} records/sec")
        print(f"  ⏱️  {ts_results['total_duration']:.2f} seconds")
        print(f"  💾 {ts_results['memory_used_mb']:.1f} MB memory")

        # Performance comparison
        if ts_results['records_per_second'] > 0 and pg_results['records_per_second'] > 0:
            if ts_results['records_per_second'] > pg_results['records_per_second']:
                speedup = ts_results['records_per_second'] / pg_results['records_per_second']
                print(f"🚀 TimescaleDB is {speedup:.1f}x faster")
            else:
                speedup = pg_results['records_per_second'] / ts_results['records_per_second']
                print(f"🚀 PostgreSQL is {speedup:.1f}x faster")
        else:
            print("⚠️  Performance comparison not available due to failed insertions")

    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"results/benchmark_results_{args.records}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simplified PostgreSQL Vector Database Population Script
High-performance version with better error handling.
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


class PostgreSQLVectorPopulator:
    def __init__(self, host: str = "localhost", port: int = 5432,
                 database: str = "vectordb", user: str = "postgres",
                 password: str = "postgres"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.vector_dim = 1024
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

    def get_next_available_id(self, count: int = 1) -> int:
        """Get the next available ID range to avoid conflicts when adding to existing data"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Use a more atomic approach by updating a counter and returning the range
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS vector_id_counter (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        last_id INTEGER NOT NULL DEFAULT 0
                    );
                """)

                # Try to insert the initial counter if it doesn't exist
                cur.execute("""
                    INSERT INTO vector_id_counter (id, last_id)
                    VALUES (1, (SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings))
                    ON CONFLICT (id) DO NOTHING;
                """)

                # Atomically get and update the counter
                cur.execute("""
                    UPDATE vector_id_counter
                    SET last_id = last_id + %s
                    WHERE id = 1
                    RETURNING last_id - %s + 1;
                """, (count, count))

                result = cur.fetchone()
                if result:
                    next_id = result[0]
                else:
                    # Fallback if something goes wrong
                    cur.execute("SELECT COALESCE(MAX(vector_id), 0) + 1 FROM vector_embeddings;")
                    next_id = cur.fetchone()[0]

                conn.commit()
                return next_id
        except Exception as e:
            print(f"Error getting next available ID: {e}")
            conn.rollback()
            # Use timestamp-based fallback to avoid conflicts
            import time
            return int(time.time() * 1000) % 1000000000  # Use timestamp-based ID
        finally:
            conn.close()

    def insert_batch(self, batch_data: List[Dict[str, Any]], conn) -> int:
        """Insert a batch of records"""
        try:
            with conn.cursor() as cur:
                # Convert data to tuples for batch insert
                batch_tuples = []
                for record in batch_data:
                    vector = self.generate_random_vector()
                    chunk_data = self.generate_chunk_data(record['id'])

                    batch_tuples.append((
                        record['id'],
                        vector,
                        chunk_data['text'],
                        json.dumps(chunk_data['metadata'])
                    ))

                # Use execute_values for much better performance
                execute_values(
                    cur,
                    """
                    INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
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
                conn.commit()
                return len(batch_tuples)

        except Exception as e:
            print(f"Error inserting batch: {e}")
            conn.rollback()
            return 0

    def populate_database(self, total_records: int = 10_000):
        """Populate the database with the specified number of records"""
        print(f"\n{'='*50}")
        print(f"POPULATING POSTGRESQL DATABASE")
        print(f"{'='*50}")
        print(f"Database: {self.database}")
        print(f"Records to insert: {total_records:,}")
        print(f"Vector dimension: {self.vector_dim}")
        print(f"Batch size: {self.batch_size}")

        # Get current count and next available ID
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                current_count = cur.fetchone()[0]
                print(f"Current records in table: {current_count:,}")
        except Exception as e:
            print(f"Error getting current count: {e}")
            current_count = 0

        # Get the next available ID range to avoid conflicts
        start_id = self.get_next_available_id(total_records)
        print(f"Starting from ID: {start_id}")
        print(f"Will add {total_records:,} new records (total will be {current_count + total_records:,})")

        start_time = time.time()
        total_batches = (total_records + self.batch_size - 1) // self.batch_size

        # Create progress bar
        pbar = tqdm(total=total_records, desc="Inserting records", unit="records")

        records_inserted = 0
        successful_batches = 0
        failed_batches = 0

        try:
            for batch_idx in range(total_batches):
                batch_start_id = start_id + (batch_idx * self.batch_size)
                current_batch_size = min(self.batch_size, total_records - (batch_idx * self.batch_size))

                # Create batch data
                batch_data = [{"id": batch_start_id + i} for i in range(current_batch_size)]

                result = self.insert_batch(batch_data, conn)
                if result > 0:
                    successful_batches += 1
                    records_inserted += result
                    pbar.update(result)
                else:
                    failed_batches += 1
        finally:
            conn.close()

        pbar.close()

        end_time = time.time()
        duration = end_time - start_time

        # Get final count
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                final_count = cur.fetchone()[0]
        except Exception as e:
            print(f"Error getting final count: {e}")
            final_count = current_count + total_records
        finally:
            conn.close()

        # Print statistics
        print(f"\n{'='*50}")
        print(f"POPULATION COMPLETED")
        print(f"{'='*50}")
        print(f"Records added: {records_inserted:,}")
        print(f"Previous count: {current_count:,}")
        print(f"Final count: {final_count:,}")
        print(f"Successful batches: {successful_batches:,}")
        print(f"Failed batches: {failed_batches:,}")
        print(f"Duration: {duration:.2f} seconds")
        if duration > 0:
            print(f"Records per second: {records_inserted / duration:,.0f}")

    def get_collection_stats(self):
        """Get and display collection statistics"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get basic stats
                cur.execute("SELECT COUNT(*) as total_points FROM vector_embeddings;")
                total_points = cur.fetchone()['total_points']

                # Get metadata stats
                cur.execute("""
                    SELECT
                        AVG(LENGTH(metadata::TEXT)) as avg_metadata_size,
                        MIN(created_at) as earliest_record,
                        MAX(created_at) as latest_record
                    FROM vector_embeddings;
                """)
                metadata_stats = cur.fetchone()

                print(f"\n{'='*50}")
                print(f"COLLECTION STATISTICS")
                print(f"{'='*50}")
                print(f"Database: {self.database}")
                print(f"Total points: {total_points:,}")
                print(f"Vector dimensions: {self.vector_dim}")
                if metadata_stats['avg_metadata_size']:
                    print(f"Average metadata size: {metadata_stats['avg_metadata_size']:.1f} characters")
                if metadata_stats['earliest_record']:
                    print(f"Earliest record: {metadata_stats['earliest_record']}")
                if metadata_stats['latest_record']:
                    print(f"Latest record: {metadata_stats['latest_record']}")
                print(f"Index type: HNSW")
                print(f"Distance metric: COSINE")

        except Exception as e:
            print(f"Error getting collection stats: {e}")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Populate PostgreSQL vector database with test data")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", default="vectordb", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    parser.add_argument("--records", type=int, default=10_000, help="Number of records to insert")
    parser.add_argument("--vector-dim", type=int, default=1024, help="Vector dimension")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inserts")
    parser.add_argument("--show-stats", action="store_true", help="Show collection stats after population")

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
    populator.batch_size = args.batch_size

    try:
        # Populate database
        populator.populate_database(total_records=args.records)

        # Show stats if requested
        if args.show_stats:
            populator.get_collection_stats()

    except KeyboardInterrupt:
        print("\nPopulation interrupted by user")
    except Exception as e:
        print(f"Error during population: {e}")
        raise


if __name__ == "__main__":
    main()
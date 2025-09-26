#!/usr/bin/env python3
"""
Record Counter Script
Counts records in vector databases: Qdrant, PostgreSQL, Milvus, and Weaviate.
By default, counts all databases. Use --qdrant-only to count only Qdrant and PostgreSQL.
"""

import argparse
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import json
import signal
import time
from contextlib import contextmanager

# Optional imports for new databases
try:
    from pymilvus import connections, Collection, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


@contextmanager
def timeout_handler(seconds):
    """Context manager for handling timeouts"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


class RecordCounter:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, 
                 postgres_host="localhost", postgres_port=5432,
                 postgres_user="postgres", postgres_password="postgres",
                 postgres_db="vectordb",
                 postgres_ts_host="localhost", postgres_ts_port=5433,
                 postgres_ts_user="postgres", postgres_ts_password="postgres",
                 postgres_ts_db="vectordb",
                 milvus_host="localhost", milvus_port="19530",
                 weaviate_host="localhost", weaviate_port="8080",
                 timeout_seconds=30):
        # Initialize with timeout configurations
        self.timeout_seconds = timeout_seconds
        self.qdrant_client = QdrantClient(
            host=qdrant_host, 
            port=qdrant_port,
            timeout=timeout_seconds
        )
        self.postgres_config = {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db,
            "connect_timeout": timeout_seconds
        }
        self.postgres_ts_config = {
            "host": postgres_ts_host,
            "port": postgres_ts_port,
            "user": postgres_ts_user,
            "password": postgres_ts_password,
            "database": postgres_ts_db,
            "connect_timeout": timeout_seconds
        }
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.weaviate_host = weaviate_host
        self.weaviate_port = weaviate_port
    
    def count_qdrant_records(self, collection_name="test_vectors"):
        """Count records in Qdrant collection with timeout"""
        try:
            with timeout_handler(self.timeout_seconds):
                collection_info = self.qdrant_client.get_collection(collection_name)
                return {
                    "success": True,
                    "count": collection_info.points_count,
                    "collection_name": collection_name,
                    "vector_dim": collection_info.config.params.vectors.size,
                    "distance_metric": collection_info.config.params.vectors.distance,
                    "indexed_vectors": collection_info.indexed_vectors_count
                }
        except TimeoutError as e:
            return {
                "success": False,
                "error": f"Timeout: {str(e)}",
                "collection_name": collection_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name
            }
    
    def count_postgres_records(self):
        """Count records in PostgreSQL table with timeout"""
        conn = None
        try:
            with timeout_handler(self.timeout_seconds):
                conn = psycopg2.connect(**self.postgres_config)
                conn.autocommit = True  # Enable autocommit to avoid transaction issues
                
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Count total records
                    cur.execute("SELECT COUNT(*) as total_count FROM vector_embeddings;")
                    total_count = cur.fetchone()['total_count']
                    
                    # Get vector dimension - try different approaches
                    vector_dim = 0
                    
                    # Try to get dimension from pgvector directly (most reliable)
                    try:
                        cur.execute("""
                            SELECT vector_dims(embedding) as vector_dim
                            FROM vector_embeddings 
                            LIMIT 1;
                        """)
                        dim_result = cur.fetchone()
                        if dim_result and dim_result['vector_dim']:
                            vector_dim = dim_result['vector_dim']
                    except Exception as e:
                        print(f"Warning: Could not get vector dimension using vector_dims: {e}")
                    
                    if vector_dim == 0:
                        # Fallback: try to get dimension from column definition
                        try:
                            cur.execute("""
                                SELECT character_maximum_length 
                                FROM information_schema.columns 
                                WHERE table_name = 'vector_embeddings' 
                                AND column_name = 'embedding';
                            """)
                            dim_result = cur.fetchone()
                            if dim_result and dim_result['character_maximum_length']:
                                vector_dim = dim_result['character_maximum_length']
                        except Exception as e:
                            print(f"Warning: Could not get vector dimension from column definition: {e}")
                    
                    # Get table size info
                    try:
                        cur.execute("""
                            SELECT 
                                pg_size_pretty(pg_total_relation_size('vector_embeddings')) as table_size,
                                pg_size_pretty(pg_relation_size('vector_embeddings')) as data_size,
                                pg_size_pretty(pg_total_relation_size('vector_embeddings') - pg_relation_size('vector_embeddings')) as index_size
                        """)
                        size_info = cur.fetchone()
                    except Exception as e:
                        print(f"Warning: Could not get table size info: {e}")
                        size_info = {
                            'table_size': 'Unknown',
                            'data_size': 'Unknown', 
                            'index_size': 'Unknown'
                        }
                    
                    return {
                        "success": True,
                        "count": total_count,
                        "vector_dim": vector_dim,
                        "table_size": size_info['table_size'],
                        "data_size": size_info['data_size'],
                        "index_size": size_info['index_size']
                    }
        except TimeoutError as e:
            return {
                "success": False,
                "error": f"Timeout: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if conn:
                conn.close()
    
    def count_postgres_ts_records(self):
        """Count records in TimescaleDB table with timeout"""
        conn = None
        try:
            with timeout_handler(self.timeout_seconds):
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True  # Enable autocommit to avoid transaction issues
                
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Count total records
                    cur.execute("SELECT COUNT(*) as total_count FROM vector_embeddings_ts;")
                    total_count = cur.fetchone()['total_count']
                    
                    # Get vector dimension - try different approaches
                    vector_dim = 0
                    
                    # Try to get dimension from pgvector directly (most reliable)
                    try:
                        cur.execute("""
                            SELECT vector_dims(embedding) as vector_dim
                            FROM vector_embeddings_ts 
                            LIMIT 1;
                        """)
                        dim_result = cur.fetchone()
                        if dim_result and dim_result['vector_dim']:
                            vector_dim = dim_result['vector_dim']
                    except Exception as e:
                        print(f"Warning: Could not get vector dimension using vector_dims: {e}")
                    
                    if vector_dim == 0:
                        # Fallback: try to get dimension from column definition
                        try:
                            cur.execute("""
                                SELECT character_maximum_length 
                                FROM information_schema.columns 
                                WHERE table_name = 'vector_embeddings_ts' 
                                AND column_name = 'embedding';
                            """)
                            dim_result = cur.fetchone()
                            if dim_result and dim_result['character_maximum_length']:
                                vector_dim = dim_result['character_maximum_length']
                        except Exception as e:
                            print(f"Warning: Could not get vector dimension from column definition: {e}")
                    
                    # Get table size info - use TimescaleDB-specific functions for hypertables
                    try:
                        # First check if this is a hypertable
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT 1 FROM timescaledb_information.hypertables 
                                WHERE hypertable_name = 'vector_embeddings_ts'
                            ) as is_hypertable;
                        """)
                        is_hypertable = cur.fetchone()['is_hypertable']
                        
                        if is_hypertable:
                            # Use TimescaleDB hypertable size functions
                            cur.execute("""
                                SELECT 
                                    pg_size_pretty(hypertable_size('vector_embeddings_ts')) as table_size,
                                    pg_size_pretty(hypertable_size('vector_embeddings_ts')) as data_size,
                                    '0 bytes' as index_size
                            """)
                        else:
                            # Use regular PostgreSQL size functions
                            cur.execute("""
                                SELECT 
                                    pg_size_pretty(pg_total_relation_size('vector_embeddings_ts')) as table_size,
                                    pg_size_pretty(pg_relation_size('vector_embeddings_ts')) as data_size,
                                    pg_size_pretty(pg_total_relation_size('vector_embeddings_ts') - pg_relation_size('vector_embeddings_ts')) as index_size
                            """)
                        size_info = cur.fetchone()
                    except Exception as e:
                        print(f"Warning: Could not get table size info: {e}")
                        size_info = {'table_size': 'N/A', 'data_size': 'N/A', 'index_size': 'N/A'}
                    
                    return {
                        "success": True,
                        "count": total_count,
                        "vector_dim": vector_dim,
                        "table_size": size_info['table_size'],
                        "data_size": size_info['data_size'],
                        "index_size": size_info['index_size']
                    }
        except TimeoutError as e:
            return {
                "success": False,
                "error": f"Timeout: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if conn:
                conn.close()
    
    def count_milvus_records(self, collection_name="test_vectors"):
        """Count records in Milvus collection - using same approach as benchmark.py"""
        if not MILVUS_AVAILABLE:
            return {
                "success": False,
                "error": "pymilvus not available. Install with: pip install pymilvus"
            }
        
        try:
            # Use the same simple approach as benchmark.py
            connections.connect(alias="count_check", host=self.milvus_host, port=self.milvus_port)
            collection = Collection(collection_name, using="count_check")
            count = collection.num_entities
            connections.disconnect("count_check")
            
            return {
                "success": True,
                "count": count,
                "collection_name": collection_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name
            }
    
    def count_weaviate_records(self, collection_name="TestVectors"):
        """Count records in Weaviate collection with timeout"""
        if not WEAVIATE_AVAILABLE:
            return {
                "success": False,
                "error": "weaviate-client not available. Install with: pip install weaviate-client"
            }
        
        try:
            with timeout_handler(self.timeout_seconds):
                # Connect to Weaviate using v4 API
                client = weaviate.connect_to_local(
                    host=self.weaviate_host,
                    port=self.weaviate_port,
                    grpc_port=50051
                )
                
                if not client.is_ready():
                    return {
                        "success": False,
                        "error": "Weaviate server not ready"
                    }
                
                # Check if collection exists
                if not client.collections.exists(collection_name):
                    return {
                        "success": False,
                        "error": f"Collection '{collection_name}' does not exist"
                    }
                
                # Get collection and count
                collection = client.collections.get(collection_name)
                result = collection.aggregate.over_all(total_count=True)
                count = result.total_count
                
                # Get collection config
                config = collection.config.get()
                
                return {
                    "success": True,
                    "count": count,
                    "collection_name": collection_name,
                    "vectorizer": "none",  # We're using custom vectors
                    "index_type": "HNSW",  # Default for Weaviate
                    "distance_metric": "COSINE"  # Default for Weaviate
                }
            
        except TimeoutError as e:
            return {
                "success": False,
                "error": f"Timeout: {str(e)}",
                "collection_name": collection_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name
            }
        finally:
            # Close Weaviate connection
            if 'client' in locals():
                client.close()
    
    
    def count_all_collections(self):
        """Count records in all Qdrant collections with timeout"""
        try:
            with timeout_handler(self.timeout_seconds):
                collections = self.qdrant_client.get_collections()
                collection_counts = {}
                
                for collection in collections.collections:
                    try:
                        collection_info = self.qdrant_client.get_collection(collection.name)
                        collection_counts[collection.name] = {
                            "count": collection_info.points_count,
                            "vector_dim": collection_info.config.params.vectors.size,
                            "distance_metric": collection_info.config.params.vectors.distance,
                            "indexed_vectors": collection_info.indexed_vectors_count
                        }
                    except Exception as e:
                        collection_counts[collection.name] = {
                            "error": str(e)
                        }
                
                return {
                    "success": True,
                    "collections": collection_counts
                }
        except TimeoutError as e:
            return {
                "success": False,
                "error": f"Timeout: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def print_summary(self, qdrant_result, postgres_result, postgres_ts_result=None, milvus_result=None, weaviate_result=None, all_collections=None):
        """Print a formatted summary of record counts"""
        print("="*80)
        print("DATABASE RECORD COUNT SUMMARY")
        print("="*80)
        
        # Qdrant results
        if qdrant_result:
            print(f"\n🔍 QDRANT DATABASE:")
            if qdrant_result["success"]:
                print(f"  Collection: {qdrant_result['collection_name']}")
                print(f"  Records: {qdrant_result['count']:,}")
                print(f"  Vector Dimension: {qdrant_result['vector_dim']}")
                print(f"  Distance Metric: {qdrant_result['distance_metric']}")
                print(f"  Indexed Vectors: {qdrant_result['indexed_vectors']:,}")
            else:
                print(f"  ❌ Error: {qdrant_result['error']}")
        
        # PostgreSQL results
        if postgres_result:
            print(f"\n🐘 POSTGRESQL DATABASE:")
            if postgres_result["success"]:
                print(f"  Table: vector_embeddings")
                print(f"  Records: {postgres_result['count']:,}")
                print(f"  Vector Dimension: {postgres_result['vector_dim']}")
                print(f"  Table Size: {postgres_result['table_size']}")
                print(f"  Data Size: {postgres_result['data_size']}")
                print(f"  Index Size: {postgres_result['index_size']}")
            else:
                print(f"  ❌ Error: {postgres_result['error']}")
        
        # TimescaleDB results
        if postgres_ts_result:
            print(f"\n⏰ TIMESCALEDB DATABASE:")
            if postgres_ts_result["success"]:
                print(f"  Table: vector_embeddings_ts (hypertable)")
                print(f"  Records: {postgres_ts_result['count']:,}")
                print(f"  Vector Dimension: {postgres_ts_result['vector_dim']}")
                print(f"  Table Size: {postgres_ts_result['table_size']}")
                print(f"  Data Size: {postgres_ts_result['data_size']}")
                print(f"  Index Size: {postgres_ts_result['index_size']}")
            else:
                print(f"  ❌ Error: {postgres_ts_result['error']}")
        
        # Milvus results
        if milvus_result:
            print(f"\n🚀 MILVUS DATABASE:")
            if milvus_result["success"]:
                print(f"  Collection: {milvus_result['collection_name']}")
                print(f"  Records: {milvus_result['count']:,}")
                # Only print additional info if available
                if 'vector_dim' in milvus_result:
                    print(f"  Vector Dimension: {milvus_result['vector_dim']}")
                if 'index_type' in milvus_result:
                    print(f"  Index Type: {milvus_result['index_type']}")
                if 'metric_type' in milvus_result:
                    print(f"  Metric Type: {milvus_result['metric_type']}")
            else:
                print(f"  ❌ Error: {milvus_result['error']}")
        
        # Weaviate results
        if weaviate_result:
            print(f"\n🔮 WEAVIATE DATABASE:")
            if weaviate_result["success"]:
                print(f"  Class: {weaviate_result['collection_name']}")
                print(f"  Records: {weaviate_result['count']:,}")
                print(f"  Vectorizer: {weaviate_result['vectorizer']}")
                print(f"  Index Type: {weaviate_result['index_type']}")
                print(f"  Distance Metric: {weaviate_result['distance_metric']}")
            else:
                print(f"  ❌ Error: {weaviate_result['error']}")
        
        # All collections if requested
        if all_collections and all_collections["success"]:
            print(f"\n📊 ALL QDRANT COLLECTIONS:")
            for name, info in all_collections["collections"].items():
                if "error" in info:
                    print(f"  {name}: ❌ {info['error']}")
                else:
                    print(f"  {name}: {info['count']:,} records ({info['vector_dim']}D)")
        
        # Multi-database comparison
        successful_results = []
        if qdrant_result and qdrant_result["success"]:
            successful_results.append(("Qdrant", qdrant_result["count"]))
        if postgres_result and postgres_result["success"]:
            successful_results.append(("PostgreSQL", postgres_result["count"]))
        if postgres_ts_result and postgres_ts_result["success"]:
            successful_results.append(("TimescaleDB", postgres_ts_result["count"]))
        if milvus_result and milvus_result["success"]:
            successful_results.append(("Milvus", milvus_result["count"]))
        if weaviate_result and weaviate_result["success"]:
            successful_results.append(("Weaviate", weaviate_result["count"]))
        
        if len(successful_results) > 1:
            print(f"\n📈 MULTI-DATABASE COMPARISON:")
            counts = [count for _, count in successful_results]
            if len(set(counts)) == 1:
                print(f"  ✅ All databases have the same number of records: {counts[0]:,}")
            else:
                print(f"  📊 Record counts across databases:")
                for db_name, count in successful_results:
                    print(f"    {db_name}: {count:,} records")
                
                max_count = max(counts)
                min_count = min(counts)
                max_db = next(name for name, count in successful_results if count == max_count)
                min_db = next(name for name, count in successful_results if count == min_count)
                diff = max_count - min_count
                print(f"  📈 {max_db} has {diff:,} more records than {min_db}")
    
    def run_count(self, collection_name="test_vectors", show_all_collections=False,
                  qdrant_only=False, postgres_only=False, timescale_only=False,
                  milvus_only=False, weaviate_only=False):
        """Run the record counting process"""
        print(f"Counting records in '{collection_name}' collection...")

        # Initialize all results as None
        qdrant_result = None
        postgres_result = None
        postgres_ts_result = None
        milvus_result = None
        weaviate_result = None

        # Count based on selected options
        if not any([qdrant_only, postgres_only, timescale_only, milvus_only, weaviate_only]):
            # Default: count all databases
            print("Counting records in all databases...")
            qdrant_result = self.count_qdrant_records(collection_name)
            postgres_result = self.count_postgres_records()
            postgres_ts_result = self.count_postgres_ts_records()
            milvus_result = self.count_milvus_records(collection_name)
            weaviate_result = self.count_weaviate_records("TestVectors")
        else:
            # Count only selected databases
            if qdrant_only:
                print("Counting records in Qdrant only...")
                qdrant_result = self.count_qdrant_records(collection_name)
            if postgres_only:
                print("Counting records in PostgreSQL only...")
                postgres_result = self.count_postgres_records()
            if timescale_only:
                print("Counting records in TimescaleDB only...")
                postgres_ts_result = self.count_postgres_ts_records()
            if milvus_only:
                print("Counting records in Milvus only...")
                milvus_result = self.count_milvus_records(collection_name)
            if weaviate_only:
                print("Counting records in Weaviate only...")
                weaviate_result = self.count_weaviate_records("TestVectors")

        # Count all collections if requested (only for Qdrant)
        all_collections = None
        if show_all_collections and (qdrant_result is not None or not any([postgres_only, timescale_only, milvus_only, weaviate_only])):
            all_collections = self.count_all_collections()

        # Print summary
        self.print_summary(qdrant_result, postgres_result, postgres_ts_result, milvus_result, weaviate_result, all_collections)

        return {
            "qdrant": qdrant_result,
            "postgres": postgres_result,
            "postgres_ts": postgres_ts_result,
            "milvus": milvus_result,
            "weaviate": weaviate_result,
            "all_collections": all_collections
        }


def main():
    parser = argparse.ArgumentParser(
        description="Count records in vector databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python count_records.py                          # Count all databases (default)
  python count_records.py --qdrant-only            # Count only Qdrant
  python count_records.py --postgres-only          # Count only PostgreSQL
  python count_records.py --timescale-only         # Count only TimescaleDB
  python count_records.py --milvus-only            # Count only Milvus
  python count_records.py --weaviate-only          # Count only Weaviate
  python count_records.py --collection my_vectors  # Specify collection name
  python count_records.py --all-collections        # Show all Qdrant collections
        """
    )

    # Collection and display options
    parser.add_argument("--collection", default="test_vectors", help="Collection/table name to count (default: test_vectors)")
    parser.add_argument("--all-collections", action="store_true", help="Show counts for all Qdrant collections")

    # Database selection options (mutually exclusive with all-databases)
    db_group = parser.add_mutually_exclusive_group()
    db_group.add_argument("--qdrant-only", action="store_true", help="Count records only in Qdrant")
    db_group.add_argument("--postgres-only", action="store_true", help="Count records only in PostgreSQL")
    db_group.add_argument("--timescale-only", action="store_true", help="Count records only in TimescaleDB")
    db_group.add_argument("--milvus-only", action="store_true", help="Count records only in Milvus")
    db_group.add_argument("--weaviate-only", action="store_true", help="Count records only in Weaviate")
    db_group.add_argument("--all", action="store_true", help="Count records in all databases (default behavior)")
    db_group.add_argument("--all-databases", action="store_true", help="Count records in all databases (this is the default behavior)")

    # Database connection arguments
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port (default: 6333)")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host (default: localhost)")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port (default: 5432)")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user (default: postgres)")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password (default: postgres)")
    parser.add_argument("--postgres-db", default="vectordb", help="PostgreSQL database (default: vectordb)")
    parser.add_argument("--postgres-ts-host", default="localhost", help="TimescaleDB host (default: localhost)")
    parser.add_argument("--postgres-ts-port", type=int, default=5433, help="TimescaleDB port (default: 5433)")
    parser.add_argument("--postgres-ts-user", default="postgres", help="TimescaleDB user (default: postgres)")
    parser.add_argument("--postgres-ts-password", default="postgres", help="TimescaleDB password (default: postgres)")
    parser.add_argument("--postgres-ts-db", default="vectordb", help="TimescaleDB database (default: vectordb)")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus host (default: localhost)")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port (default: 19530)")
    parser.add_argument("--weaviate-host", default="localhost", help="Weaviate host (default: localhost)")
    parser.add_argument("--weaviate-port", type=int, default=8080, help="Weaviate port (default: 8080)")
    
    # Timeout configuration
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for database operations (default: 30)")

    args = parser.parse_args()

    counter = RecordCounter(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        postgres_db=args.postgres_db,
        postgres_ts_host=args.postgres_ts_host,
        postgres_ts_port=args.postgres_ts_port,
        postgres_ts_user=args.postgres_ts_user,
        postgres_ts_password=args.postgres_ts_password,
        postgres_ts_db=args.postgres_ts_db,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        weaviate_host=args.weaviate_host,
        weaviate_port=args.weaviate_port,
        timeout_seconds=args.timeout
    )

    try:
        results = counter.run_count(
            collection_name=args.collection,
            show_all_collections=args.all_collections,
            qdrant_only=args.qdrant_only,
            postgres_only=args.postgres_only,
            timescale_only=args.timescale_only,
            milvus_only=args.milvus_only,
            weaviate_only=args.weaviate_only
        )
    except KeyboardInterrupt:
        print("\nCount interrupted by user")
    except Exception as e:
        print(f"Error during count: {e}")
        raise


if __name__ == "__main__":
    main()

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


class RecordCounter:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, 
                 postgres_host="localhost", postgres_port=5432,
                 postgres_user="postgres", postgres_password="postgres",
                 postgres_db="vectordb",
                 postgres_ts_host="localhost", postgres_ts_port=5433,
                 postgres_ts_user="postgres", postgres_ts_password="postgres",
                 postgres_ts_db="vectordb",
                 milvus_host="localhost", milvus_port="19530",
                 weaviate_host="localhost", weaviate_port="8080"):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.postgres_config = {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db
        }
        self.postgres_ts_config = {
            "host": postgres_ts_host,
            "port": postgres_ts_port,
            "user": postgres_ts_user,
            "password": postgres_ts_password,
            "database": postgres_ts_db
        }
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.weaviate_host = weaviate_host
        self.weaviate_port = weaviate_port
    
    def count_qdrant_records(self, collection_name="test_vectors"):
        """Count records in Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            return {
                "success": True,
                "count": collection_info.points_count,
                "collection_name": collection_name,
                "vector_dim": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "indexed_vectors": collection_info.indexed_vectors_count
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name
            }
    
    def count_postgres_records(self):
        """Count records in PostgreSQL table"""
        conn = None
        try:
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
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if conn:
                conn.close()
    
    def count_postgres_ts_records(self):
        """Count records in TimescaleDB table"""
        conn = None
        try:
            conn = psycopg2.connect(**self.postgres_ts_config)
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
                    size_info = {'table_size': 'N/A', 'data_size': 'N/A', 'index_size': 'N/A'}
                
                return {
                    "success": True,
                    "count": total_count,
                    "vector_dim": vector_dim,
                    "table_size": size_info['table_size'],
                    "data_size": size_info['data_size'],
                    "index_size": size_info['index_size']
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
        """Count records in Milvus collection"""
        if not MILVUS_AVAILABLE:
            return {
                "success": False,
                "error": "pymilvus not available. Install with: pip install pymilvus"
            }
        
        try:
            # Connect to Milvus
            connection_alias = f"milvus_count_{collection_name}"
            connections.connect(
                alias=connection_alias,
                host=self.milvus_host,
                port=self.milvus_port
            )
            
            # Check if collection exists
            if not utility.has_collection(collection_name, using=connection_alias):
                return {
                    "success": False,
                    "error": f"Collection '{collection_name}' does not exist"
                }
            
            # Get collection info
            collection = Collection(collection_name, using=connection_alias)
            collection.load()
            
            # Get entity count
            entity_count = collection.num_entities
            
            # Get schema info
            schema = collection.schema
            vector_dim = 0
            for field in schema.fields:
                if field.name == "vector":
                    vector_dim = field.params.get("dim", 0)
                    break
            
            # Get index info
            index_info = collection.index().params
            
            connections.disconnect(connection_alias)
            
            return {
                "success": True,
                "count": entity_count,
                "collection_name": collection_name,
                "vector_dim": vector_dim,
                "index_type": index_info.get("index_type", "Unknown"),
                "metric_type": index_info.get("metric_type", "Unknown")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name
            }
    
    def count_weaviate_records(self, collection_name="TestVectors"):
        """Count records in Weaviate collection"""
        if not WEAVIATE_AVAILABLE:
            return {
                "success": False,
                "error": "weaviate-client not available. Install with: pip install weaviate-client"
            }
        
        try:
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
        """Count records in all Qdrant collections"""
        try:
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
                print(f"  Table: vector_embeddings (hypertable)")
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
                print(f"  Vector Dimension: {milvus_result['vector_dim']}")
                print(f"  Index Type: {milvus_result['index_type']}")
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
        if qdrant_result["success"]:
            successful_results.append(("Qdrant", qdrant_result["count"]))
        if postgres_result["success"]:
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
    
    def run_count(self, collection_name="test_vectors", show_all_collections=False, include_all_databases=True, qdrant_only=False):
        """Run the record counting process"""
        print(f"Counting records in '{collection_name}' collection...")
        
        # Count specific collection
        qdrant_result = self.count_qdrant_records(collection_name)
        postgres_result = self.count_postgres_records()
        postgres_ts_result = self.count_postgres_ts_records()
        
        # Count other databases based on flags
        milvus_result = None
        weaviate_result = None
        
        if include_all_databases and not qdrant_only:
            print("Counting records in all databases...")
            milvus_result = self.count_milvus_records(collection_name)
            weaviate_result = self.count_weaviate_records("TestVectors")
        elif qdrant_only:
            print("Counting records in Qdrant and PostgreSQL only...")
        
        # Count all collections if requested
        all_collections = None
        if show_all_collections:
            all_collections = self.count_all_collections()
        
        # Print summary
        self.print_summary(qdrant_result, postgres_result, postgres_ts_result, milvus_result, weaviate_result, all_collections)
        
        return {
            "qdrant": qdrant_result,
            "postgres": postgres_result,
            "postgres_ts": postgres_ts_result,
            "all_collections": all_collections
        }


def main():
    parser = argparse.ArgumentParser(description="Count records in vector databases")
    parser.add_argument("--collection", default="test_vectors", help="Collection name to count")
    parser.add_argument("--all-collections", action="store_true", help="Show counts for all Qdrant collections")
    parser.add_argument("--all-databases", action="store_true", default=True, help="Count records in all databases (Qdrant, PostgreSQL, Milvus, Weaviate) - this is the default behavior")
    parser.add_argument("--qdrant-only", action="store_true", help="Count records only in Qdrant and PostgreSQL (skip Milvus and Weaviate)")
    
    # Database connection arguments
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--postgres-db", default="vectordb", help="PostgreSQL database")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--weaviate-host", default="localhost", help="Weaviate host")
    parser.add_argument("--weaviate-port", type=int, default=8080, help="Weaviate port")
    
    args = parser.parse_args()
    
    counter = RecordCounter(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        postgres_db=args.postgres_db,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        weaviate_host=args.weaviate_host,
        weaviate_port=args.weaviate_port,
    )
    
    try:
        results = counter.run_count(
            collection_name=args.collection,
            show_all_collections=args.all_collections,
            include_all_databases=args.all_databases,
            qdrant_only=args.qdrant_only
        )
    except KeyboardInterrupt:
        print("\nCount interrupted by user")
    except Exception as e:
        print(f"Error during count: {e}")
        raise


if __name__ == "__main__":
    main()

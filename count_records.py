#!/usr/bin/env python3
"""
Record Counter Script
Counts records in both Qdrant and PostgreSQL databases for the test_vectors collection.
"""

import argparse
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import RealDictCursor


class RecordCounter:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, 
                 postgres_host="localhost", postgres_port=5432,
                 postgres_user="postgres", postgres_password="postgres",
                 postgres_db="vectordb"):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.postgres_config = {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db
        }
    
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
    
    def print_summary(self, qdrant_result, postgres_result, all_collections=None):
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
        
        # All collections if requested
        if all_collections and all_collections["success"]:
            print(f"\n📊 ALL QDRANT COLLECTIONS:")
            for name, info in all_collections["collections"].items():
                if "error" in info:
                    print(f"  {name}: ❌ {info['error']}")
                else:
                    print(f"  {name}: {info['count']:,} records ({info['vector_dim']}D)")
        
        # Comparison
        if qdrant_result["success"] and postgres_result["success"]:
            print(f"\n📈 COMPARISON:")
            qdrant_count = qdrant_result["count"]
            postgres_count = postgres_result["count"]
            
            if qdrant_count == postgres_count:
                print(f"  ✅ Both databases have the same number of records: {qdrant_count:,}")
            else:
                diff = abs(qdrant_count - postgres_count)
                if qdrant_count > postgres_count:
                    print(f"  📊 Qdrant has {diff:,} more records than PostgreSQL")
                else:
                    print(f"  📊 PostgreSQL has {diff:,} more records than Qdrant")
                
                print(f"  Qdrant: {qdrant_count:,} records")
                print(f"  PostgreSQL: {postgres_count:,} records")
    
    def run_count(self, collection_name="test_vectors", show_all_collections=False):
        """Run the record counting process"""
        print(f"Counting records in '{collection_name}' collection...")
        
        # Count specific collection
        qdrant_result = self.count_qdrant_records(collection_name)
        postgres_result = self.count_postgres_records()
        
        # Count all collections if requested
        all_collections = None
        if show_all_collections:
            all_collections = self.count_all_collections()
        
        # Print summary
        self.print_summary(qdrant_result, postgres_result, all_collections)
        
        return {
            "qdrant": qdrant_result,
            "postgres": postgres_result,
            "all_collections": all_collections
        }


def main():
    parser = argparse.ArgumentParser(description="Count records in vector databases")
    parser.add_argument("--collection", default="test_vectors", help="Qdrant collection name to count")
    parser.add_argument("--all-collections", action="store_true", help="Show counts for all Qdrant collections")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--postgres-db", default="vectordb", help="PostgreSQL database")
    
    args = parser.parse_args()
    
    counter = RecordCounter(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        postgres_db=args.postgres_db
    )
    
    try:
        results = counter.run_count(
            collection_name=args.collection,
            show_all_collections=args.all_collections
        )
    except KeyboardInterrupt:
        print("\nCount interrupted by user")
    except Exception as e:
        print(f"Error during count: {e}")
        raise


if __name__ == "__main__":
    main()

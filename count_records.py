#!/usr/bin/env python3
"""
Record Counter Script
Counts records in all vector databases: Qdrant, PostgreSQL, Milvus, Weaviate, and Vespa.
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
                 milvus_host="localhost", milvus_port="19530",
                 weaviate_host="localhost", weaviate_port="8080",
                 vespa_host="localhost", vespa_port="8081"):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.postgres_config = {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db
        }
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.weaviate_host = weaviate_host
        self.weaviate_port = weaviate_port
        self.vespa_host = vespa_host
        self.vespa_port = vespa_port
    
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
            if not utility.has_collection(collection_name):
                return {
                    "success": False,
                    "error": f"Collection '{collection_name}' does not exist"
                }
            
            # Get collection info
            collection = Collection(collection_name)
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
            # Connect to Weaviate
            client = weaviate.Client(
                url=f"http://{self.weaviate_host}:{self.weaviate_port}",
                additional_headers={"X-OpenAI-Api-Key": "dummy"}
            )
            
            if not client.is_ready():
                return {
                    "success": False,
                    "error": "Weaviate server not ready"
                }
            
            # Check if class exists
            if not client.schema.exists(collection_name):
                return {
                    "success": False,
                    "error": f"Class '{collection_name}' does not exist"
                }
            
            # Get object count
            result = client.query.aggregate(collection_name).with_meta_count().do()
            if "data" in result and "Aggregate" in result["data"]:
                count = result["data"]["Aggregate"][collection_name][0]["meta"]["count"]
            else:
                count = 0
            
            # Get class schema
            schema = client.schema.get(collection_name)
            vectorizer_config = schema.get("vectorizer", {})
            vector_index_config = vectorizer_config.get("vectorIndexConfig", {})
            
            return {
                "success": True,
                "count": count,
                "collection_name": collection_name,
                "vectorizer": vectorizer_config.get("vectorizer", "none"),
                "index_type": vector_index_config.get("vectorIndexType", "Unknown"),
                "distance_metric": vector_index_config.get("distance", "Unknown")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "collection_name": collection_name
            }
    
    def count_vespa_records(self, application_name="test_vectors", document_type="test_vector"):
        """Count records in Vespa application"""
        try:
            # Test connection
            response = requests.get(f"http://{self.vespa_host}:{self.vespa_port}/ApplicationStatus", timeout=10)
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Vespa server not ready (status: {response.status_code})"
                }
            
            # Count documents using search
            search_url = f"http://{self.vespa_host}:{self.vespa_port}/search/"
            params = {
                "yql": f"select * from {document_type} limit 0",
                "hits": 0
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                total_count = data.get("root", {}).get("totalCount", 0)
            else:
                return {
                    "success": False,
                    "error": f"Could not query Vespa (status: {response.status_code})"
                }
            
            return {
                "success": True,
                "count": total_count,
                "application_name": application_name,
                "document_type": document_type,
                "search_endpoint": f"http://{self.vespa_host}:{self.vespa_port}/search/"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "application_name": application_name
            }
    
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
    
    def print_summary(self, qdrant_result, postgres_result, milvus_result=None, weaviate_result=None, vespa_result=None, all_collections=None):
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
        
        # Vespa results
        if vespa_result:
            print(f"\n⚡ VESPA DATABASE:")
            if vespa_result["success"]:
                print(f"  Application: {vespa_result['application_name']}")
                print(f"  Document Type: {vespa_result['document_type']}")
                print(f"  Records: {vespa_result['count']:,}")
                print(f"  Search Endpoint: {vespa_result['search_endpoint']}")
            else:
                print(f"  ❌ Error: {vespa_result['error']}")
        
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
        if milvus_result and milvus_result["success"]:
            successful_results.append(("Milvus", milvus_result["count"]))
        if weaviate_result and weaviate_result["success"]:
            successful_results.append(("Weaviate", weaviate_result["count"]))
        if vespa_result and vespa_result["success"]:
            successful_results.append(("Vespa", vespa_result["count"]))
        
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
    
    def run_count(self, collection_name="test_vectors", show_all_collections=False, include_all_databases=False):
        """Run the record counting process"""
        print(f"Counting records in '{collection_name}' collection...")
        
        # Count specific collection
        qdrant_result = self.count_qdrant_records(collection_name)
        postgres_result = self.count_postgres_records()
        
        # Count other databases if requested
        milvus_result = None
        weaviate_result = None
        vespa_result = None
        
        if include_all_databases:
            print("Counting records in all databases...")
            milvus_result = self.count_milvus_records(collection_name)
            weaviate_result = self.count_weaviate_records("TestVectors")
            vespa_result = self.count_vespa_records("test_vectors", "test_vector")
        
        # Count all collections if requested
        all_collections = None
        if show_all_collections:
            all_collections = self.count_all_collections()
        
        # Print summary
        self.print_summary(qdrant_result, postgres_result, milvus_result, weaviate_result, vespa_result, all_collections)
        
        return {
            "qdrant": qdrant_result,
            "postgres": postgres_result,
            "all_collections": all_collections
        }


def main():
    parser = argparse.ArgumentParser(description="Count records in vector databases")
    parser.add_argument("--collection", default="test_vectors", help="Collection name to count")
    parser.add_argument("--all-collections", action="store_true", help="Show counts for all Qdrant collections")
    parser.add_argument("--all-databases", action="store_true", help="Count records in all databases (Qdrant, PostgreSQL, Milvus, Weaviate, Vespa)")
    
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
    parser.add_argument("--vespa-host", default="localhost", help="Vespa host")
    parser.add_argument("--vespa-port", type=int, default=8081, help="Vespa port")
    
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
        vespa_host=args.vespa_host,
        vespa_port=args.vespa_port
    )
    
    try:
        results = counter.run_count(
            collection_name=args.collection,
            show_all_collections=args.all_collections,
            include_all_databases=args.all_databases
        )
    except KeyboardInterrupt:
        print("\nCount interrupted by user")
    except Exception as e:
        print(f"Error during count: {e}")
        raise


if __name__ == "__main__":
    main()

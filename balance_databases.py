#!/usr/bin/env python3
"""
Database Balancing Script
Counts records in all vector databases and adds new records to make all databases have equal counts.
Rounds up to the next thousand of the highest count for fair performance comparisons.
"""

import argparse
import time
import numpy as np
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import psycopg2
from psycopg2.extras import RealDictCursor
import weaviate
import weaviate.classes as wvc

# Optional imports for Milvus
try:
    from pymilvus import connections, Collection, utility, DataType, FieldSchema, CollectionSchema
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class DatabaseBalancer:
    def __init__(self, 
                 qdrant_host="localhost", qdrant_port=6333,
                 postgres_host="localhost", postgres_port=5432,
                 postgres_user="postgres", postgres_password="postgres",
                 postgres_db="vectordb",
                 postgres_ts_host="localhost", postgres_ts_port=5433,
                 postgres_ts_user="postgres", postgres_ts_password="postgres",
                 postgres_ts_db="vectordb",
                 milvus_host="localhost", milvus_port=19530,
                 weaviate_host="localhost", weaviate_port=8080):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
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
        
        self.vector_dim = 1024  # Default vector dimension
        self.target_count = 0
        
    def get_vector_dimension(self) -> int:
        """Get vector dimension from existing data"""
        try:
            # Try Qdrant first
            collections = self.qdrant_client.get_collections()
            if collections.collections:
                collection_info = self.qdrant_client.get_collection(collections.collections[0].name)
                return collection_info.config.params.vectors.size
        except Exception:
            pass
            
        try:
            # Try PostgreSQL as fallback
            conn = psycopg2.connect(**self.postgres_config)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT array_length(embedding::float[], 1) 
                    FROM vector_embeddings 
                    LIMIT 1;
                """)
                result = cur.fetchone()
                if result and result[0]:
                    return result[0]
        except Exception:
            pass
            
        print("Warning: Could not determine vector dimension, using default 1024")
        return 1024
    
    def generate_vector(self) -> List[float]:
        """Generate a random normalized vector"""
        vector = np.random.normal(0, 1, self.vector_dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    def count_qdrant_records(self, collection_name: str = "test_vectors") -> Dict[str, Any]:
        """Count records in Qdrant"""
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            return {
                "success": True,
                "count": collection_info.points_count,
                "collection_name": collection_name,
                "vector_dim": self.vector_dim
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def count_postgres_records(self) -> Dict[str, Any]:
        """Count records in PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                count = cur.fetchone()[0]
                
                # Get vector dimension by parsing the vector string
                cur.execute("SELECT embedding FROM vector_embeddings LIMIT 1;")
                result = cur.fetchone()
                if result and result[0]:
                    # Parse the vector string to get dimension
                    vector_str = result[0]
                    if vector_str.startswith('[') and vector_str.endswith(']'):
                        vector_list = eval(vector_str)  # Convert string representation to list
                        vector_dim = len(vector_list)
                    else:
                        vector_dim = self.vector_dim
                else:
                    vector_dim = self.vector_dim
                
            conn.close()
            return {
                "success": True,
                "count": count,
                "vector_dim": vector_dim
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def count_postgres_ts_records(self) -> Dict[str, Any]:
        """Count records in TimescaleDB"""
        try:
            conn = psycopg2.connect(**self.postgres_ts_config)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings_ts;")
                count = cur.fetchone()[0]
                
                # Get vector dimension by parsing the vector string
                cur.execute("SELECT embedding FROM vector_embeddings_ts LIMIT 1;")
                result = cur.fetchone()
                if result and result[0]:
                    # Parse the vector string to get dimension
                    vector_str = result[0]
                    if vector_str.startswith('[') and vector_str.endswith(']'):
                        vector_list = eval(vector_str)  # Convert string representation to list
                        vector_dim = len(vector_list)
                    else:
                        vector_dim = self.vector_dim
                else:
                    vector_dim = self.vector_dim
                
            conn.close()
            return {
                "success": True,
                "count": count,
                "vector_dim": vector_dim
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def count_milvus_records(self, collection_name: str = "test_vectors") -> Dict[str, Any]:
        """Count records in Milvus"""
        if not MILVUS_AVAILABLE:
            return {"success": False, "error": "pymilvus not available"}
        
        try:
            connection_alias = f"milvus_balance_{collection_name}"
            connections.connect(
                alias=connection_alias,
                host=self.milvus_host,
                port=self.milvus_port
            )
            
            if not utility.has_collection(collection_name, using=connection_alias):
                return {"success": False, "error": f"Collection '{collection_name}' does not exist"}
            
            collection = Collection(collection_name, using=connection_alias)
            collection.load()
            
            # Get collection stats
            stats = collection.num_entities
            
            connections.disconnect(connection_alias)
            
            return {
                "success": True,
                "count": stats,
                "collection_name": collection_name,
                "vector_dim": self.vector_dim
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def count_weaviate_records(self, class_name: str = "TestVectors") -> Dict[str, Any]:
        """Count records in Weaviate"""
        try:
            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port
            )
            
            if not client.collections.exists(class_name):
                return {"success": False, "error": f"Class '{class_name}' does not exist"}
            
            collection = client.collections.get(class_name)
            result = collection.aggregate.over_all(total_count=True)
            count = result.total_count
            
            client.close()
            
            return {
                "success": True,
                "count": count,
                "collection_name": class_name,
                "vector_dim": self.vector_dim
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def add_qdrant_records(self, collection_name: str, count: int) -> bool:
        """Add records to Qdrant"""
        try:
            print(f"Adding {count:,} records to Qdrant...")
            
            # Generate points in batches
            batch_size = 1000
            for i in range(0, count, batch_size):
                current_batch_size = min(batch_size, count - i)
                points = []
                
                for j in range(current_batch_size):
                    point_id = int(time.time() * 1000000) + i + j  # Unique ID
                    vector = self.generate_vector()
                    payload = {
                        "id": point_id,
                        "text": f"Balanced point {point_id}",
                        "metadata": {
                            "category": "balanced",
                            "value": np.random.uniform(0, 100),
                            "timestamp": int(time.time())
                        }
                    }
                    
                    points.append(PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    ))
                
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                if (i + batch_size) % 10000 == 0:
                    print(f"  Added {i + batch_size:,} records...")
            
            print(f"✅ Successfully added {count:,} records to Qdrant")
            return True
        except Exception as e:
            print(f"❌ Error adding records to Qdrant: {e}")
            return False
    
    def add_postgres_records(self, count: int) -> bool:
        """Add records to PostgreSQL"""
        try:
            print(f"Adding {count:,} records to PostgreSQL...")
            
            conn = psycopg2.connect(**self.postgres_config)
            
            # Get current max ID
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                max_id = cur.fetchone()[0]
            
            # Generate records in batches
            batch_size = 1000
            for i in range(0, count, batch_size):
                current_batch_size = min(batch_size, count - i)
                
                values = []
                for j in range(current_batch_size):
                    vector_id = max_id + 200000 + i + j  # Use large offset
                    vector = self.generate_vector()
                    text_content = f"Balanced document {vector_id}"
                    metadata = f'{{"source": "balanced", "id": {vector_id}}}'
                    
                    values.append(f"({vector_id}, ARRAY{vector}::vector, '{text_content}', '{metadata}')")
                
                insert_query = f"""
                    INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                    VALUES {', '.join(values)};
                """
                
                with conn.cursor() as cur:
                    cur.execute(insert_query)
                    conn.commit()
                
                if (i + batch_size) % 10000 == 0:
                    print(f"  Added {i + batch_size:,} records...")
            
            conn.close()
            print(f"✅ Successfully added {count:,} records to PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Error adding records to PostgreSQL: {e}")
            return False
    
    def add_postgres_ts_records(self, count: int) -> bool:
        """Add records to TimescaleDB"""
        try:
            print(f"Adding {count:,} records to TimescaleDB...")
            
            conn = psycopg2.connect(**self.postgres_ts_config)
            
            # Get current max ID
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings_ts;")
                max_id = cur.fetchone()[0]
            
            # Generate records in batches
            batch_size = 1000
            for i in range(0, count, batch_size):
                current_batch_size = min(batch_size, count - i)
                
                values = []
                for j in range(current_batch_size):
                    vector_id = max_id + 200000 + i + j  # Use large offset
                    vector = self.generate_vector()
                    text_content = f"Balanced TimescaleDB document {vector_id}"
                    metadata = f'{{"source": "balanced_timescaledb", "id": {vector_id}}}'
                    
                    values.append(f"({vector_id}, ARRAY{vector}::vector, '{text_content}', '{metadata}')")
                
                insert_query = f"""
                    INSERT INTO vector_embeddings_ts (vector_id, embedding, text_content, metadata)
                    VALUES {', '.join(values)};
                """
                
                with conn.cursor() as cur:
                    cur.execute(insert_query)
                    conn.commit()
                
                if (i + batch_size) % 10000 == 0:
                    print(f"  Added {i + batch_size:,} records...")
            
            conn.close()
            print(f"✅ Successfully added {count:,} records to TimescaleDB")
            return True
        except Exception as e:
            print(f"❌ Error adding records to TimescaleDB: {e}")
            return False
    
    def add_milvus_records(self, collection_name: str, count: int) -> bool:
        """Add records to Milvus"""
        if not MILVUS_AVAILABLE:
            print("❌ pymilvus not available")
            return False
        
        try:
            print(f"Adding {count:,} records to Milvus...")
            
            connection_alias = f"milvus_balance_{collection_name}"
            connections.connect(
                alias=connection_alias,
                host=self.milvus_host,
                port=self.milvus_port
            )
            
            collection = Collection(collection_name, using=connection_alias)
            
            # Generate records in batches
            batch_size = 1000
            for i in range(0, count, batch_size):
                current_batch_size = min(batch_size, count - i)
                
                vectors = []
                text_contents = []
                metadata_list = []
                
                for j in range(current_batch_size):
                    vectors.append(self.generate_vector())
                    text_contents.append(f"Balanced document {int(time.time() * 1000000) + i + j}")
                    metadata_list.append(f'{{"source": "balanced", "id": {int(time.time() * 1000000) + i + j}}}')
                
                data = [vectors, text_contents, metadata_list]
                collection.insert(data)
                collection.flush()
                
                if (i + batch_size) % 10000 == 0:
                    print(f"  Added {i + batch_size:,} records...")
            
            connections.disconnect(connection_alias)
            print(f"✅ Successfully added {count:,} records to Milvus")
            return True
        except Exception as e:
            print(f"❌ Error adding records to Milvus: {e}")
            return False
    
    def add_weaviate_records(self, class_name: str, count: int) -> bool:
        """Add records to Weaviate"""
        try:
            print(f"Adding {count:,} records to Weaviate...")
            
            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port
            )
            
            collection = client.collections.get(class_name)
            
            # Generate records in batches
            batch_size = 1000
            for i in range(0, count, batch_size):
                current_batch_size = min(batch_size, count - i)
                
                data_objects = []
                for j in range(current_batch_size):
                    vector = self.generate_vector()
                    doc_id = f"balanced_{int(time.time() * 1000000) + i + j}"
                    
                    data_objects.append(wvc.data.DataObject(
                        properties={
                            "text_content": f"Balanced document {doc_id}",
                            "metadata": f'{{"source": "balanced", "id": "{doc_id}"}}'
                        },
                        vector=vector
                    ))
                
                collection.data.insert_many(data_objects)
                
                if (i + batch_size) % 10000 == 0:
                    print(f"  Added {i + batch_size:,} records...")
            
            client.close()
            print(f"✅ Successfully added {count:,} records to Weaviate")
            return True
        except Exception as e:
            print(f"❌ Error adding records to Weaviate: {e}")
            return False
    
    def balance_databases(self, 
                         qdrant_collection: str = "test_vectors",
                         milvus_collection: str = "test_vectors", 
                         weaviate_class: str = "TestVectors",
                         force_balance: bool = False,
                         dry_run: bool = False):
        """Balance all databases to have equal record counts"""
        
        print("="*80)
        print("DATABASE BALANCING SCRIPT")
        print("="*80)
        
        # Get vector dimension
        self.vector_dim = self.get_vector_dimension()
        print(f"Vector Dimension: {self.vector_dim}")
        
        # Count records in all databases
        print("\n📊 Counting records in all databases...")
        
        qdrant_result = self.count_qdrant_records(qdrant_collection)
        postgres_result = self.count_postgres_records()
        postgres_ts_result = self.count_postgres_ts_records()
        milvus_result = self.count_milvus_records(milvus_collection)
        weaviate_result = self.count_weaviate_records(weaviate_class)
        
        # Display current counts
        print("\nCurrent Record Counts:")
        print("-" * 40)
        
        counts = {}
        if qdrant_result["success"]:
            counts["Qdrant"] = qdrant_result["count"]
            print(f"🔍 Qdrant:     {qdrant_result['count']:,} records")
        else:
            print(f"❌ Qdrant:     Error - {qdrant_result['error']}")
        
        if postgres_result["success"]:
            counts["PostgreSQL"] = postgres_result["count"]
            print(f"🐘 PostgreSQL: {postgres_result['count']:,} records")
        else:
            print(f"❌ PostgreSQL: Error - {postgres_result['error']}")
        
        if postgres_ts_result["success"]:
            counts["TimescaleDB"] = postgres_ts_result["count"]
            print(f"⏰ TimescaleDB: {postgres_ts_result['count']:,} records")
        else:
            print(f"❌ TimescaleDB: Error - {postgres_ts_result['error']}")
        
        if milvus_result["success"]:
            counts["Milvus"] = milvus_result["count"]
            print(f"🚀 Milvus:     {milvus_result['count']:,} records")
        else:
            print(f"❌ Milvus:     Error - {milvus_result['error']}")
        
        if weaviate_result["success"]:
            counts["Weaviate"] = weaviate_result["count"]
            print(f"🔮 Weaviate:   {weaviate_result['count']:,} records")
        else:
            print(f"❌ Weaviate:   Error - {weaviate_result['error']}")
        
        if not counts:
            print("❌ No databases available for balancing")
            return
        
        # Calculate target count
        max_count = max(counts.values())
        self.target_count = ((max_count // 1000) + 1) * 1000  # Round up to next thousand
        
        print(f"\n🎯 Target Count: {self.target_count:,} records (rounded up from {max_count:,})")
        
        # Check if balancing is needed
        if not force_balance and all(count == self.target_count for count in counts.values()):
            print("✅ All databases already have the target count!")
            return
        
        # Calculate records to add for each database
        records_to_add = {}
        for db_name, current_count in counts.items():
            needed = self.target_count - current_count
            if needed > 0:
                records_to_add[db_name] = needed
                print(f"📈 {db_name}: needs {needed:,} more records")
            else:
                print(f"✅ {db_name}: already has target count")
        
        if not records_to_add:
            print("✅ All databases are already balanced!")
            return
        
        # Handle dry run
        if dry_run:
            total_to_add = sum(records_to_add.values())
            print(f"\n🔍 DRY RUN: Would add {total_to_add:,} total records across all databases.")
            print("Use --force to actually perform the balancing.")
            return
        
        # Confirm before proceeding
        if not force_balance:
            total_to_add = sum(records_to_add.values())
            print(f"\n⚠️  This will add {total_to_add:,} total records across all databases.")
            try:
                response = input("Do you want to proceed? (y/N): ").strip().lower()
                if response != 'y':
                    print("❌ Operation cancelled")
                    return
            except EOFError:
                print("❌ Operation cancelled (no input available)")
                return
        
        # Add records to each database
        print(f"\n🔄 Adding records to balance databases...")
        
        success_count = 0
        
        if "Qdrant" in records_to_add and qdrant_result["success"]:
            if self.add_qdrant_records(qdrant_collection, records_to_add["Qdrant"]):
                success_count += 1
        
        if "PostgreSQL" in records_to_add and postgres_result["success"]:
            if self.add_postgres_records(records_to_add["PostgreSQL"]):
                success_count += 1
        
        if "TimescaleDB" in records_to_add and postgres_ts_result["success"]:
            if self.add_postgres_ts_records(records_to_add["TimescaleDB"]):
                success_count += 1
        
        if "Milvus" in records_to_add and milvus_result["success"]:
            if self.add_milvus_records(milvus_collection, records_to_add["Milvus"]):
                success_count += 1
        
        if "Weaviate" in records_to_add and weaviate_result["success"]:
            if self.add_weaviate_records(weaviate_class, records_to_add["Weaviate"]):
                success_count += 1
        
        # Final verification
        print(f"\n🔍 Verifying final counts...")
        final_qdrant = self.count_qdrant_records(qdrant_collection)
        final_postgres = self.count_postgres_records()
        final_postgres_ts = self.count_postgres_ts_records()
        final_milvus = self.count_milvus_records(milvus_collection)
        final_weaviate = self.count_weaviate_records(weaviate_class)
        
        print("\nFinal Record Counts:")
        print("-" * 40)
        
        final_counts = {}
        if final_qdrant["success"]:
            final_counts["Qdrant"] = final_qdrant["count"]
            print(f"🔍 Qdrant:     {final_qdrant['count']:,} records")
        
        if final_postgres["success"]:
            final_counts["PostgreSQL"] = final_postgres["count"]
            print(f"🐘 PostgreSQL: {final_postgres['count']:,} records")
        
        if final_postgres_ts["success"]:
            final_counts["TimescaleDB"] = final_postgres_ts["count"]
            print(f"⏰ TimescaleDB: {final_postgres_ts['count']:,} records")
        
        if final_milvus["success"]:
            final_counts["Milvus"] = final_milvus["count"]
            print(f"🚀 Milvus:     {final_milvus['count']:,} records")
        
        if final_weaviate["success"]:
            final_counts["Weaviate"] = final_weaviate["count"]
            print(f"🔮 Weaviate:   {final_weaviate['count']:,} records")
        
        # Check if all databases are balanced
        if final_counts:
            all_balanced = all(count == self.target_count for count in final_counts.values())
            if all_balanced:
                print(f"\n✅ SUCCESS: All databases now have {self.target_count:,} records!")
            else:
                print(f"\n⚠️  WARNING: Some databases may not be perfectly balanced")
                print(f"Target: {self.target_count:,} records")
                for db_name, count in final_counts.items():
                    if count != self.target_count:
                        print(f"  {db_name}: {count:,} records (diff: {count - self.target_count:+,})")
        
        print(f"\n📊 Summary: Successfully balanced {success_count} out of {len(records_to_add)} databases")


def main():
    parser = argparse.ArgumentParser(description="Balance record counts across all vector databases")
    
    # Database connection options
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--postgres-db", default="vectordb", help="PostgreSQL database")
    parser.add_argument("--postgres-ts-host", default="localhost", help="TimescaleDB host")
    parser.add_argument("--postgres-ts-port", type=int, default=5433, help="TimescaleDB port")
    parser.add_argument("--postgres-ts-user", default="postgres", help="TimescaleDB user")
    parser.add_argument("--postgres-ts-password", default="postgres", help="TimescaleDB password")
    parser.add_argument("--postgres-ts-db", default="vectordb", help="TimescaleDB database")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--weaviate-host", default="localhost", help="Weaviate host")
    parser.add_argument("--weaviate-port", type=int, default=8080, help="Weaviate port")
    
    # Collection options
    parser.add_argument("--qdrant-collection", default="test_vectors", help="Qdrant collection name")
    parser.add_argument("--milvus-collection", default="test_vectors", help="Milvus collection name")
    parser.add_argument("--weaviate-class", default="TestVectors", help="Weaviate class name")
    
    # Control options
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    balancer = DatabaseBalancer(
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
        weaviate_port=args.weaviate_port
    )
    
    try:
        balancer.balance_databases(
            qdrant_collection=args.qdrant_collection,
            milvus_collection=args.milvus_collection,
            weaviate_class=args.weaviate_class,
            force_balance=args.force,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error during balancing: {e}")
        raise


if __name__ == "__main__":
    main()

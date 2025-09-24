#!/usr/bin/env python3
"""
Comprehensive Vector Database Benchmark Suite
Consolidates all benchmark functionality into a single script with flags for different test types.
"""

import time
import psutil
import os
import json
import argparse
import statistics
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchRequest, QueryRequest
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
import requests

# Optional imports for new databases
try:
    from pymilvus import connections, Collection, utility, DataType, FieldSchema, CollectionSchema
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

try:
    import weaviate
    import weaviate.classes as wvc
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


class ComprehensiveBenchmarkSuite:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, 
                 postgres_host="localhost", postgres_port=5432,
                 postgres_user="postgres", postgres_password="postgres",
                 postgres_db="vectordb",
                 postgres_ts_host="localhost", postgres_ts_port=5433,
                 postgres_ts_user="postgres", postgres_ts_password="postgres",
                 postgres_ts_db="vectordb",
                 milvus_host="localhost", milvus_port=19530,
                 weaviate_host="localhost", weaviate_port=8080,
                 ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_client = QdrantClient(
            host=qdrant_host, 
            port=qdrant_port,
            timeout=60.0  # 60 second timeout
        )
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
        self.vector_dim = self._get_vector_dimension()
        self.results = {}
        self.next_id = 0
        
    def _get_vector_dimension(self) -> int:
        """Get vector dimension from Qdrant collection or PostgreSQL table"""
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
    
    def get_system_info(self):
        """Get system information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent,
            "python_version": os.sys.version,
            "platform": os.name
        }
    
    def generate_query_vector(self) -> List[float]:
        """Generate a random normalized query vector"""
        vector = np.random.normal(0, 1, self.vector_dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    def generate_test_point(self, point_id: int) -> PointStruct:
        """Generate a test point for write operations"""
        vector = self.generate_query_vector()
        payload = {
            "id": point_id,
            "text": f"Test point {point_id}",
            "metadata": {
                "category": np.random.choice(["A", "B", "C", "D"]),
                "value": np.random.uniform(0, 100),
                "timestamp": int(time.time())
            }
        }
        
        return PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
    
    # ==================== READ BENCHMARKS ====================
    
    def run_read_benchmark(self, collection_name: str, iterations: int = 100):
        """Run comprehensive read performance benchmark"""
        print(f"\n{'='*60}")
        print(f"READ BENCHMARK - Collection: {collection_name}")
        print(f"{'='*60}")
        
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            print(f"Collection: {collection_name}")
            print(f"Total points: {collection_info.points_count:,}")
            print(f"Vector dimension: {self.vector_dim}")
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None
        
        read_results = {}
        
        # 1. Single Vector Search
        print("\n1. Single Vector Search (10 results)")
        single_search_times = []
        for _ in tqdm(range(iterations), desc="Single searches"):
            query_vector = self.generate_query_vector()
            start_time = time.time()
            self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=10
            )
            single_search_times.append(time.time() - start_time)
        
        read_results['single_search'] = {
            'mean': statistics.mean(single_search_times),
            'median': statistics.median(single_search_times),
            'p95': np.percentile(single_search_times, 95),
            'p99': np.percentile(single_search_times, 99),
            'min': min(single_search_times),
            'max': max(single_search_times),
            'qps': 1.0 / statistics.mean(single_search_times)
        }
        
        # 2. Batch Search
        print("\n2. Batch Vector Search (5 vectors, 10 results each)")
        batch_search_times = []
        batch_iterations = max(1, iterations // 10)
        for _ in tqdm(range(batch_iterations), desc="Batch searches"):
            query_vectors = [self.generate_query_vector() for _ in range(5)]  # Reduced from 10 to 5
            start_time = time.time()
            self.qdrant_client.query_batch_points(
                collection_name=collection_name,
                requests=[QueryRequest(query=v, limit=10) for v in query_vectors]
            )
            batch_search_times.append(time.time() - start_time)
        
        read_results['batch_search'] = {
            'mean': statistics.mean(batch_search_times),
            'median': statistics.median(batch_search_times),
            'p95': np.percentile(batch_search_times, 95),
            'p99': np.percentile(batch_search_times, 99),
            'min': min(batch_search_times),
            'max': max(batch_search_times),
            'qps': 1.0 / statistics.mean(batch_search_times)
        }
        
        # 3. Filtered Search
        print("\n3. Filtered Vector Search")
        filtered_search_times = []
        categories = ["A", "B", "C", "D"]
        for _ in tqdm(range(iterations), desc="Filtered searches"):
            query_vector = self.generate_query_vector()
            random_category = np.random.choice(categories)
            
            start_time = time.time()
            self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[FieldCondition(
                        key="metadata.category",
                        match=MatchValue(value=random_category)
                    )]
                ),
                limit=10
            )
            filtered_search_times.append(time.time() - start_time)
        
        read_results['filtered_search'] = {
            'mean': statistics.mean(filtered_search_times),
            'median': statistics.median(filtered_search_times),
            'p95': np.percentile(filtered_search_times, 95),
            'p99': np.percentile(filtered_search_times, 99),
            'min': min(filtered_search_times),
            'max': max(filtered_search_times),
            'qps': 1.0 / statistics.mean(filtered_search_times)
        }
        
        # 4. Retrieve by ID
        print("\n4. Retrieve by ID (10 points)")
        retrieve_times = []
        point_ids = [np.random.randint(0, collection_info.points_count) for _ in range(10)]
        for _ in tqdm(range(iterations), desc="ID retrievals"):
            start_time = time.time()
            self.qdrant_client.retrieve(
                collection_name=collection_name,
                ids=point_ids
            )
            retrieve_times.append(time.time() - start_time)
        
        read_results['retrieve_by_id'] = {
            'mean': statistics.mean(retrieve_times),
            'median': statistics.median(retrieve_times),
            'p95': np.percentile(retrieve_times, 95),
            'p99': np.percentile(retrieve_times, 99),
            'min': min(retrieve_times),
            'max': max(retrieve_times),
            'qps': 1.0 / statistics.mean(retrieve_times)
        }
        
        # 5. Scroll
        print("\n5. Scroll Collection (1000 points)")
        scroll_times = []
        scroll_iterations = max(1, iterations // 10)
        for _ in tqdm(range(scroll_iterations), desc="Scroll operations"):
            start_time = time.time()
            self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            scroll_times.append(time.time() - start_time)
        
        read_results['scroll'] = {
            'mean': statistics.mean(scroll_times),
            'median': statistics.median(scroll_times),
            'p95': np.percentile(scroll_times, 95),
            'p99': np.percentile(scroll_times, 99),
            'min': min(scroll_times),
            'max': max(scroll_times),
            'qps': 1.0 / statistics.mean(scroll_times)
        }
        
        # 6. Concurrent Searches
        concurrent_queries = max(10, iterations)  # At least 10 queries for meaningful concurrent testing
        print(f"\n6. Concurrent Searches ({concurrent_queries} queries, 10 workers)")
        def single_search():
            query_vector = self.generate_query_vector()
            start_time = time.time()
            self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=10
            )
            return time.time() - start_time
        
        concurrent_times = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(single_search) for _ in range(concurrent_queries)]
            for future in tqdm(as_completed(futures), total=concurrent_queries, desc="Concurrent searches"):
                concurrent_times.append(future.result())
        
        read_results['concurrent_search'] = {
            'mean': statistics.mean(concurrent_times),
            'median': statistics.median(concurrent_times),
            'p95': np.percentile(concurrent_times, 95),
            'p99': np.percentile(concurrent_times, 99),
            'min': min(concurrent_times),
            'max': max(concurrent_times),
            'qps': 1.0 / statistics.mean(concurrent_times)
        }
        
        return read_results
    
    # ==================== WRITE BENCHMARKS ====================
    
    def run_write_benchmark(self, collection_name: str, iterations: int = 100, cleanup: bool = True):
        """Run comprehensive write performance benchmark"""
        print(f"\n{'='*60}")
        print(f"WRITE BENCHMARK - Collection: {collection_name}")
        print(f"{'='*60}")
        
        # Check if collection exists
        try:
            collections = self.qdrant_client.get_collections()
            if any(col.name == collection_name for col in collections.collections):
                print(f"Using existing collection '{collection_name}'")
            else:
                print(f"Error: Collection '{collection_name}' does not exist. Please run populate script first.")
                return None
        except Exception as e:
            print(f"Error checking collection: {e}")
            return None
        
        write_results = {}
        self.next_id = 0
        
        # 1. Single Point Insertions
        print("\n1. Single Point Insertions")
        single_insert_times = []
        for _ in tqdm(range(iterations), desc="Single inserts"):
            point = self.generate_test_point(self.next_id)
            self.next_id += 1
            
            start_time = time.time()
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            single_insert_times.append(time.time() - start_time)
        
        write_results['single_insert'] = {
            'mean': statistics.mean(single_insert_times),
            'median': statistics.median(single_insert_times),
            'p95': np.percentile(single_insert_times, 95),
            'p99': np.percentile(single_insert_times, 99),
            'min': min(single_insert_times),
            'max': max(single_insert_times),
            'throughput': 1.0 / statistics.mean(single_insert_times)
        }
        
        # 2. Batch Insertions
        batch_sizes = [10, 100, 1000]
        for batch_size in batch_sizes:
            print(f"\n2. Batch Insertions (batch size: {batch_size})")
            batch_insert_times = []
            num_batches = max(1, iterations // batch_size)
            
            for _ in tqdm(range(num_batches), desc=f"Batch {batch_size}"):
                points = [self.generate_test_point(self.next_id + i) for i in range(batch_size)]
                self.next_id += batch_size
                
                start_time = time.time()
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                batch_insert_times.append(time.time() - start_time)
            
            write_results[f'batch_insert_{batch_size}'] = {
                'mean': statistics.mean(batch_insert_times),
                'median': statistics.median(batch_insert_times),
                'p95': np.percentile(batch_insert_times, 95),
                'p99': np.percentile(batch_insert_times, 99),
                'min': min(batch_insert_times),
                'max': max(batch_insert_times),
                'batch_size': batch_size,
                'throughput': batch_size / statistics.mean(batch_insert_times)
            }
        
        # 3. Concurrent Insertions
        concurrent_batches = max(5, iterations // 2)  # At least 5 batches for meaningful concurrent testing
        batch_size = 100
        print(f"\n3. Concurrent Insertions (10 workers, {concurrent_batches} batches, batch size {batch_size})")
        def batch_insert():
            points = [self.generate_test_point(self.next_id + i) for i in range(batch_size)]
            self.next_id += batch_size
            start_time = time.time()
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            return time.time() - start_time
        
        concurrent_times = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(batch_insert) for _ in range(concurrent_batches)]
            for future in tqdm(as_completed(futures), total=concurrent_batches, desc="Concurrent inserts"):
                concurrent_times.append(future.result())
        
        write_results['concurrent_insert'] = {
            'mean': statistics.mean(concurrent_times),
            'median': statistics.median(concurrent_times),
            'p95': np.percentile(concurrent_times, 95),
            'p99': np.percentile(concurrent_times, 99),
            'min': min(concurrent_times),
            'max': max(concurrent_times),
            'throughput': 1.0 / statistics.mean(concurrent_times)
        }
        
        # 4. Update Operations
        print("\n4. Update Operations")
        update_times = []
        for _ in tqdm(range(iterations), desc="Update operations"):
            point_id = np.random.randint(0, self.next_id - 1)
            point = self.generate_test_point(point_id)
            
            start_time = time.time()
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            update_times.append(time.time() - start_time)
        
        write_results['update'] = {
            'mean': statistics.mean(update_times),
            'median': statistics.median(update_times),
            'p95': np.percentile(update_times, 95),
            'p99': np.percentile(update_times, 99),
            'min': min(update_times),
            'max': max(update_times),
            'throughput': 1.0 / statistics.mean(update_times)
        }
        
        # 5. Delete Operations
        print("\n5. Delete Operations")
        delete_times = []
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            point_ids_to_delete = [np.random.randint(0, collection_info.points_count - 1) for _ in range(min(iterations, 100))]
        except:
            point_ids_to_delete = list(range(min(iterations, 100)))
        
        for point_id in tqdm(point_ids_to_delete, desc="Delete operations"):
            start_time = time.time()
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=[point_id]
            )
            delete_times.append(time.time() - start_time)
        
        write_results['delete'] = {
            'mean': statistics.mean(delete_times),
            'median': statistics.median(delete_times),
            'p95': np.percentile(delete_times, 95),
            'p99': np.percentile(delete_times, 99),
            'min': min(delete_times),
            'max': max(delete_times),
            'throughput': 1.0 / statistics.mean(delete_times)
        }
        
        # Note: Collection is preserved for future use
        
        return write_results
    
    # ==================== POSTGRESQL BENCHMARKS ====================
    
    def run_postgres_benchmark(self, iterations: int = 100):
        """Run PostgreSQL search and insert benchmarks"""
        print(f"\n{'='*60}")
        print(f"POSTGRESQL BENCHMARK")
        print(f"{'='*60}")
        
        postgres_results = {}
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            
            # Get PostgreSQL stats
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                count = cur.fetchone()[0]
                print(f"PostgreSQL Points: {count:,}")
                print(f"Vector Dimension: {self.vector_dim}")
            
            # Search performance
            print("\n1. PostgreSQL Search Performance")
            search_times = []
            for _ in tqdm(range(iterations), desc="PostgreSQL searches"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM vector_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, query_vector))
                    cur.fetchall()
                search_times.append(time.time() - start_time)
            
            postgres_results['single_search'] = {
                'mean': statistics.mean(search_times),
                'median': statistics.median(search_times),
                'p95': np.percentile(search_times, 95),
                'p99': np.percentile(search_times, 99),
                'min': min(search_times),
                'max': max(search_times),
                'qps': 1.0 / statistics.mean(search_times)
            }
            
            # Batch search performance
            print("\n2. PostgreSQL Batch Search Performance")
            batch_search_times = []
            batch_iterations = max(1, iterations // 10)
            for _ in tqdm(range(batch_iterations), desc="PostgreSQL batch searches"):
                query_vectors = [self.generate_query_vector() for _ in range(10)]
                start_time = time.time()
                with conn.cursor() as cur:
                    for query_vector in query_vectors:
                        cur.execute("""
                            SELECT vector_id, text_content, metadata,
                                   1 - (embedding <=> %s::vector) AS similarity
                            FROM vector_embeddings
                            ORDER BY embedding <=> %s::vector
                            LIMIT 10;
                        """, (query_vector, query_vector))
                        cur.fetchall()
                batch_search_times.append(time.time() - start_time)
            
            postgres_results['batch_search'] = {
                'mean': statistics.mean(batch_search_times),
                'qps': (10 * len(batch_search_times)) / sum(batch_search_times)
            }
            
            # Filtered search performance
            print("\n3. PostgreSQL Filtered Search Performance")
            filtered_search_times = []
            for _ in tqdm(range(iterations), desc="PostgreSQL filtered searches"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM vector_embeddings
                        WHERE vector_id > %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, np.random.randint(1000, 50000), query_vector))
                    cur.fetchall()
                filtered_search_times.append(time.time() - start_time)
            
            postgres_results['filtered_search'] = {
                'mean': statistics.mean(filtered_search_times),
                'qps': 1.0 / statistics.mean(filtered_search_times)
            }
            
            # ID retrieval performance
            print("\n4. PostgreSQL ID Retrieval Performance")
            id_retrieval_times = []
            for _ in tqdm(range(iterations), desc="PostgreSQL ID retrievals"):
                # Get random IDs from the database
                with conn.cursor() as cur:
                    cur.execute("SELECT vector_id FROM vector_embeddings ORDER BY RANDOM() LIMIT 10;")
                    random_ids = [row[0] for row in cur.fetchall()]
                
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata, embedding
                        FROM vector_embeddings
                        WHERE vector_id = ANY(%s);
                    """, (random_ids,))
                    cur.fetchall()
                id_retrieval_times.append(time.time() - start_time)
            
            postgres_results['retrieve_by_id'] = {
                'mean': statistics.mean(id_retrieval_times),
                'qps': 1.0 / statistics.mean(id_retrieval_times)
            }
            
            # Concurrent search performance
            print(f"\n5. PostgreSQL Concurrent Search Performance ({concurrent_queries} queries, 10 workers)")
            concurrent_search_times = []
            def postgres_search_worker():
                query_vector = self.generate_query_vector()
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM vector_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, query_vector))
                    cur.fetchall()
                return time.time() - start_time
            
            concurrent_queries = max(10, iterations)  # At least 10 queries for meaningful concurrent testing
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(postgres_search_worker) for _ in range(concurrent_queries)]
                concurrent_search_times = [future.result() for future in futures]
            
            postgres_results['concurrent_search'] = {
                'mean': statistics.mean(concurrent_search_times),
                'qps': 1.0 / statistics.mean(concurrent_search_times)
            }
            
            # Insert performance
            print("\n6. PostgreSQL Insert Performance")
            insert_times = []
            
            # Get the current max ID to avoid conflicts
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                max_id = cur.fetchone()[0]
            
            for i in tqdm(range(iterations), desc="PostgreSQL inserts"):
                vector = self.generate_query_vector()
                unique_id = max_id + 100000 + i  # Use a large offset to avoid conflicts
                
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                        VALUES (%s, %s::vector, %s, %s);
                    """, (unique_id, vector, f"Test point {i}", '{"test": true}'))
                    conn.commit()
                insert_times.append(time.time() - start_time)
            
            postgres_results['single_insert'] = {
                'mean': statistics.mean(insert_times),
                'median': statistics.median(insert_times),
                'p95': np.percentile(insert_times, 95),
                'p99': np.percentile(insert_times, 99),
                'min': min(insert_times),
                'max': max(insert_times),
                'throughput': 1.0 / statistics.mean(insert_times)
            }
            
            # Batch insert performance
            print("\n7. PostgreSQL Batch Insert Performance")
            batch_insert_times = []
            
            # Get the current max ID to avoid conflicts
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                max_id = cur.fetchone()[0]
            
            batch_iterations = max(1, iterations // 10)
            for i in tqdm(range(batch_iterations), desc="PostgreSQL batch inserts"):
                start_time = time.time()
                with conn.cursor() as cur:
                    # Insert 100 records in a single transaction
                    values = []
                    for j in range(100):
                        vector = self.generate_query_vector()
                        unique_id = max_id + 200000 + (i * 100) + j
                        values.append(f"({unique_id}, ARRAY{vector}::vector, 'Batch point {i}-{j}', '{{\"batch\": true}}')")
                    
                    cur.execute(f"""
                        INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                        VALUES {', '.join(values)};
                    """)
                    conn.commit()
                batch_insert_times.append(time.time() - start_time)
            
            postgres_results['batch_insert_100'] = {
                'mean': statistics.mean(batch_insert_times),
                'throughput': (100 * len(batch_insert_times)) / sum(batch_insert_times)
            }
            
            # Update performance
            print("\n8. PostgreSQL Update Performance")
            update_times = []
            
            for i in tqdm(range(iterations), desc="PostgreSQL updates"):
                # Get a random existing record
                with conn.cursor() as cur:
                    cur.execute("SELECT vector_id FROM vector_embeddings ORDER BY RANDOM() LIMIT 1;")
                    record_id = cur.fetchone()[0]
                
                new_vector = self.generate_query_vector()
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE vector_embeddings 
                        SET embedding = %s::vector, text_content = %s, metadata = %s
                        WHERE vector_id = %s;
                    """, (new_vector, f"Updated point {i}", '{"updated": true}', record_id))
                    conn.commit()
                update_times.append(time.time() - start_time)
            
            postgres_results['update'] = {
                'mean': statistics.mean(update_times),
                'throughput': 1.0 / statistics.mean(update_times)
            }
            
            # Delete performance
            print("\n9. PostgreSQL Delete Performance")
            delete_times = []
            
            for i in tqdm(range(iterations), desc="PostgreSQL deletes"):
                # Get a random existing record
                with conn.cursor() as cur:
                    cur.execute("SELECT vector_id FROM vector_embeddings ORDER BY RANDOM() LIMIT 1;")
                    record_id = cur.fetchone()[0]
                
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM vector_embeddings WHERE vector_id = %s;", (record_id,))
                    conn.commit()
                delete_times.append(time.time() - start_time)
            
            postgres_results['delete'] = {
                'mean': statistics.mean(delete_times),
                'throughput': 1.0 / statistics.mean(delete_times)
            }
            
            conn.close()
            
        except Exception as e:
            print(f"PostgreSQL benchmark error: {e}")
            return None
        
        return postgres_results
    
    # ==================== COMPARISON BENCHMARKS ====================
    
    def run_database_comparison(self, qdrant_collection: str, iterations: int = 100):
        """Run direct comparison between Qdrant and PostgreSQL"""
        print(f"\n{'='*60}")
        print(f"DATABASE COMPARISON")
        print(f"{'='*60}")
        
        # Use lighter iterations for comparison to avoid timeouts
        comparison_iterations = min(iterations, 10)
        
        # Run Qdrant benchmark with reduced iterations
        try:
            qdrant_read = self.run_read_benchmark(qdrant_collection, comparison_iterations)
            qdrant_write = self.run_write_benchmark(qdrant_collection, comparison_iterations, cleanup=False)
        except Exception as e:
            print(f"❌ Qdrant benchmark failed: {e}")
            qdrant_read = qdrant_write = None
        
        # Run PostgreSQL benchmark with reduced iterations
        try:
            postgres_results = self.run_postgres_benchmark(comparison_iterations)
        except Exception as e:
            print(f"❌ PostgreSQL benchmark failed: {e}")
            postgres_results = None
        
        if not any([qdrant_read, postgres_results]):
            print("Error: Could not complete database comparison - all benchmarks failed")
            return None
        
        # Calculate performance ratios
        comparison_results = {
            'qdrant_read': qdrant_read,
            'qdrant_write': qdrant_write,
            'postgres': postgres_results,
            'ratios': {}
        }
        
        # Search comparison
        if qdrant_read and 'single_search' in qdrant_read and postgres_results and 'single_search' in postgres_results:
            qdrant_qps = qdrant_read['single_search']['qps']
            postgres_qps = postgres_results['single_search']['qps']
            if postgres_qps > 0:
                comparison_results['ratios']['search_qps'] = {
                    'qdrant_vs_postgres': qdrant_qps / postgres_qps,
                    'qdrant_qps': qdrant_qps,
                    'postgres_qps': postgres_qps
                }
        
        # Insert comparison
        if qdrant_write and 'single_insert' in qdrant_write and postgres_results and 'single_insert' in postgres_results:
            qdrant_throughput = qdrant_write['single_insert']['throughput']
            postgres_throughput = postgres_results['single_insert']['throughput']
            if postgres_throughput > 0:
                comparison_results['ratios']['insert_throughput'] = {
                    'qdrant_vs_postgres': qdrant_throughput / postgres_throughput,
                    'qdrant_throughput': qdrant_throughput,
                    'postgres_throughput': postgres_throughput
                }
        
        return comparison_results
    
    # ==================== LOAD TEST ====================
    
    def run_load_test(self, collection_name: str, write_collection_name: str, duration: int = 120):
        """Run load test for specified duration"""
        print(f"\n{'='*60}")
        print(f"LOAD TEST ({duration} seconds) - Collection: {collection_name}")
        print(f"{'='*60}")
        
        import threading
        
        def continuous_reads():
            """Continuously perform read operations"""
            while not stop_event.is_set():
                try:
                    query_vector = self.generate_query_vector()
                    self.qdrant_client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        limit=10
                    )
                except Exception as e:
                    print(f"Error in continuous reads: {e}")
        
        def continuous_writes():
            """Continuously perform write operations"""
            # Use the existing write collection for load testing
            write_collection = write_collection_name
            try:
                while not stop_event.is_set():
                    points = [self.generate_test_point(self.next_id + i) for i in range(100)]
                    self.next_id += 100
                    self.qdrant_client.upsert(
                        collection_name=write_collection,
                        points=points
                    )
            except Exception as e:
                print(f"Error in continuous writes: {e}")
        
        stop_event = threading.Event()
        
        # Monitor system resources
        print(f"Monitoring system for {duration} seconds...")
        cpu_usage = []
        memory_usage = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=1))
            memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(1)
        
        # Start load test
        with ThreadPoolExecutor(max_workers=4) as executor:
            read_future = executor.submit(continuous_reads)
            write_future = executor.submit(continuous_writes)
            
            # Wait for duration
            time.sleep(duration)
            
            # Stop all operations
            stop_event.set()
            
            # Wait for threads to finish
            read_future.result(timeout=10)
            write_future.result(timeout=10)
        
        return {
            "cpu_usage": {
                "mean": sum(cpu_usage) / len(cpu_usage),
                "max": max(cpu_usage),
                "min": min(cpu_usage)
            },
            "memory_usage": {
                "mean": sum(memory_usage) / len(memory_usage),
                "max": max(memory_usage),
                "min": min(memory_usage)
            },
            "duration": duration
        }
    
    # ==================== TimescaleDB BENCHMARKS ====================
    
    def run_postgres_ts_benchmark(self, iterations: int = 100):
        """Run TimescaleDB benchmark (read and write operations)"""
        print(f"\n{'='*60}")
        print("TIMESCALEDB BENCHMARK")
        print(f"{'='*60}")
        
        # Get current record count
        try:
            conn = psycopg2.connect(**self.postgres_ts_config)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                postgres_ts_points = cur.fetchone()[0]
            conn.close()
        except Exception as e:
            print(f"❌ Error getting TimescaleDB record count: {e}")
            return {"error": str(e)}
        
        print(f"TimescaleDB Points: {postgres_ts_points:,}")
        print(f"Vector Dimension: {self.vector_dim}")
        
        # Generate test vectors
        test_vector = np.random.rand(self.vector_dim).astype(np.float32).tolist()
        test_vectors = [np.random.rand(self.vector_dim).astype(np.float32).tolist() for _ in range(10)]
        
        # 1. TimescaleDB Search Performance
        print(f"\n1. TimescaleDB Search Performance")
        search_times = []
        for i in tqdm(range(iterations), desc="TimescaleDB searches"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata, 
                               1 - (embedding <=> %s::vector) as similarity
                        FROM vector_embeddings 
                        ORDER BY embedding <=> %s::vector 
                        LIMIT 10;
                    """, (test_vector, test_vector))
                    results = cur.fetchall()
                conn.close()
                search_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB search error: {e}")
                search_times.append(float('inf'))
        
        # 2. TimescaleDB Batch Search Performance
        print(f"\n2. TimescaleDB Batch Search Performance")
        batch_search_times = []
        batch_iterations = max(1, iterations // 10)
        for i in tqdm(range(batch_iterations), desc="TimescaleDB batch searches"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Execute 10 searches in a single query
                    for _ in range(10):
                        cur.execute("""
                            SELECT vector_id, text_content, metadata, 
                                   1 - (embedding <=> %s::vector) as similarity
                            FROM vector_embeddings 
                            ORDER BY embedding <=> %s::vector 
                            LIMIT 10;
                        """, (test_vector, test_vector))
                        results = cur.fetchall()
                conn.close()
                batch_search_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB batch search error: {e}")
                batch_search_times.append(float('inf'))
        
        # 3. TimescaleDB Filtered Search Performance
        print(f"\n3. TimescaleDB Filtered Search Performance")
        filtered_search_times = []
        for i in tqdm(range(iterations), desc="TimescaleDB filtered searches"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata, 
                               1 - (embedding <=> %s::vector) as similarity
                        FROM vector_embeddings 
                        WHERE vector_id > %s
                        ORDER BY embedding <=> %s::vector 
                        LIMIT 10;
                    """, (test_vector, 1000, test_vector))
                    results = cur.fetchall()
                conn.close()
                filtered_search_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB filtered search error: {e}")
                filtered_search_times.append(float('inf'))
        
        # 4. TimescaleDB ID Retrieval Performance
        print(f"\n4. TimescaleDB ID Retrieval Performance")
        id_retrieval_times = []
        for i in tqdm(range(iterations), desc="TimescaleDB ID retrievals"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Get random IDs first
                    cur.execute("SELECT vector_id FROM vector_embeddings ORDER BY RANDOM() LIMIT 10;")
                    random_ids = [row[0] for row in cur.fetchall()]
                    
                    # Retrieve by IDs
                    if random_ids:
                        cur.execute("""
                            SELECT vector_id, text_content, metadata 
                            FROM vector_embeddings 
                            WHERE vector_id = ANY(%s);
                        """, (random_ids,))
                        results = cur.fetchall()
                conn.close()
                id_retrieval_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB ID retrieval error: {e}")
                id_retrieval_times.append(float('inf'))
        
        # 5. TimescaleDB Concurrent Search Performance
        print(f"\n5. TimescaleDB Concurrent Search Performance ({concurrent_queries} queries, 10 workers)")
        concurrent_search_times = []
        
        def timescaledb_search_worker():
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata, 
                               1 - (embedding <=> %s::vector) as similarity
                        FROM vector_embeddings 
                        ORDER BY embedding <=> %s::vector 
                        LIMIT 10;
                    """, (test_vector, test_vector))
                    results = cur.fetchall()
                conn.close()
                return True
            except Exception as e:
                return False
        
        concurrent_queries = max(10, iterations)  # At least 10 queries for meaningful concurrent testing
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(timescaledb_search_worker) for _ in range(concurrent_queries)]
            for future in tqdm(as_completed(futures), total=concurrent_queries, desc="TimescaleDB concurrent searches"):
                start_time = time.time()
                success = future.result()
                concurrent_search_times.append(time.time() - start_time)
        
        # 6. TimescaleDB Insert Performance
        print(f"\n6. TimescaleDB Insert Performance")
        insert_times = []
        for i in tqdm(range(iterations), desc="TimescaleDB inserts"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Get max ID to avoid conflicts
                    cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                    max_id = cur.fetchone()[0]
                    
                    new_vector = np.random.rand(self.vector_dim).astype(np.float32).tolist()
                    cur.execute("""
                        INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                        VALUES (%s, %s::vector, %s, %s);
                    """, (max_id + 100000 + i, new_vector, f"TimescaleDB test document {i}", 
                          json.dumps({"source": "timescaledb_benchmark", "iteration": i})))
                conn.close()
                insert_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB insert error: {e}")
                insert_times.append(float('inf'))
        
        # 7. TimescaleDB Batch Insert Performance
        print(f"\n7. TimescaleDB Batch Insert Performance")
        batch_insert_times = []
        batch_iterations = max(1, iterations // 10)
        for i in tqdm(range(batch_iterations), desc="TimescaleDB batch inserts"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Get max ID to avoid conflicts
                    cur.execute("SELECT COALESCE(MAX(vector_id), 0) FROM vector_embeddings;")
                    max_id = cur.fetchone()[0]
                    
                    # Prepare batch data
                    batch_data = []
                    for j in range(100):
                        new_vector = np.random.rand(self.vector_dim).astype(np.float32).tolist()
                        batch_data.append((
                            max_id + 100000 + (i * 100) + j,
                            new_vector,
                            f"TimescaleDB batch document {i}-{j}",
                            json.dumps({"source": "timescaledb_batch", "batch": i, "index": j})
                        ))
                    
                    # Batch insert
                    cur.executemany("""
                        INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                        VALUES (%s, %s::vector, %s, %s);
                    """, batch_data)
                conn.close()
                batch_insert_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB batch insert error: {e}")
                batch_insert_times.append(float('inf'))
        
        # 8. TimescaleDB Update Performance
        print(f"\n8. TimescaleDB Update Performance")
        update_times = []
        for i in tqdm(range(iterations), desc="TimescaleDB updates"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Get random existing record
                    cur.execute("SELECT vector_id FROM vector_embeddings ORDER BY RANDOM() LIMIT 1;")
                    result = cur.fetchone()
                    if result:
                        vector_id = result[0]
                        new_vector = np.random.rand(self.vector_dim).astype(np.float32).tolist()
                        cur.execute("""
                            UPDATE vector_embeddings 
                            SET embedding = %s::vector, text_content = %s, metadata = %s
                            WHERE vector_id = %s;
                        """, (new_vector, f"Updated TimescaleDB document {i}", 
                              json.dumps({"source": "timescaledb_update", "iteration": i}), vector_id))
                conn.close()
                update_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB update error: {e}")
                update_times.append(float('inf'))
        
        # 9. TimescaleDB Delete Performance
        print(f"\n9. TimescaleDB Delete Performance")
        delete_times = []
        for i in tqdm(range(iterations), desc="TimescaleDB deletes"):
            start_time = time.time()
            try:
                conn = psycopg2.connect(**self.postgres_ts_config)
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Get random existing record
                    cur.execute("SELECT vector_id FROM vector_embeddings ORDER BY RANDOM() LIMIT 1;")
                    result = cur.fetchone()
                    if result:
                        vector_id = result[0]
                        cur.execute("DELETE FROM vector_embeddings WHERE vector_id = %s;", (vector_id,))
                conn.close()
                delete_times.append(time.time() - start_time)
            except Exception as e:
                print(f"❌ TimescaleDB delete error: {e}")
                delete_times.append(float('inf'))
        
        # Calculate performance metrics
        def safe_mean(times):
            valid_times = [t for t in times if t != float('inf')]
            return statistics.mean(valid_times) if valid_times else 0
        
        def safe_qps(times):
            valid_times = [t for t in times if t != float('inf')]
            return 1 / statistics.mean(valid_times) if valid_times else 0
        
        return {
            "single_search": {
                "times": search_times,
                "mean_time": safe_mean(search_times),
                "qps": safe_qps(search_times)
            },
            "batch_search": {
                "times": batch_search_times,
                "mean_time": safe_mean(batch_search_times),
                "qps": safe_qps(batch_search_times)
            },
            "filtered_search": {
                "times": filtered_search_times,
                "mean_time": safe_mean(filtered_search_times),
                "qps": safe_qps(filtered_search_times)
            },
            "retrieve_by_id": {
                "times": id_retrieval_times,
                "mean_time": safe_mean(id_retrieval_times),
                "qps": safe_qps(id_retrieval_times)
            },
            "concurrent_search": {
                "times": concurrent_search_times,
                "mean_time": safe_mean(concurrent_search_times),
                "qps": safe_qps(concurrent_search_times)
            },
            "single_insert": {
                "times": insert_times,
                "mean_time": safe_mean(insert_times),
                "throughput": safe_qps(insert_times)
            },
            "batch_insert_100": {
                "times": batch_insert_times,
                "mean_time": safe_mean(batch_insert_times),
                "throughput": safe_qps(batch_insert_times)
            },
            "update": {
                "times": update_times,
                "mean_time": safe_mean(update_times),
                "throughput": safe_qps(update_times)
            },
            "delete": {
                "times": delete_times,
                "mean_time": safe_mean(delete_times),
                "throughput": safe_qps(delete_times)
            }
        }
    
    def run_milvus_benchmark(self, collection_name: str, iterations: int = 100):
        """Run Milvus benchmark (read and write operations)"""
        if not MILVUS_AVAILABLE:
            return {"error": "pymilvus not available. Install with: pip install pymilvus"}
        
        try:
            # Connect to Milvus
            connection_alias = f"milvus_benchmark_{collection_name}"
            connections.connect(
                alias=connection_alias,
                host=self.milvus_host,
                port=self.milvus_port
            )
            
            # Check if collection exists
            if not utility.has_collection(collection_name, using=connection_alias):
                return {"error": f"Collection '{collection_name}' does not exist"}
            
            collection = Collection(collection_name, using=connection_alias)
            collection.load()
            
            # Single search benchmark
            print("\n1. Milvus Single Search Performance")
            single_search_times = []
            for _ in tqdm(range(iterations), desc="Milvus single searches"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                # Search in Milvus
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=10
                )
                
                end_time = time.time()
                single_search_times.append(end_time - start_time)
            
            # Batch search benchmark
            print("\n2. Milvus Batch Search Performance")
            batch_search_times = []
            batch_iterations = max(1, iterations // 10)
            for _ in tqdm(range(batch_iterations), desc="Milvus batch searches"):
                query_vectors = [self.generate_query_vector() for _ in range(10)]
                start_time = time.time()
                
                # Batch search in Milvus
                results = collection.search(
                    data=query_vectors,
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=10
                )
                
                end_time = time.time()
                batch_search_times.append(end_time - start_time)
            
            # Filtered search benchmark
            print("\n3. Milvus Filtered Search Performance")
            filtered_search_times = []
            for _ in tqdm(range(iterations), desc="Milvus filtered searches"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                # Search with filter in Milvus
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=10,
                    expr="id > 1000"  # Use 'id' field instead of 'vector_id'
                )
                
                end_time = time.time()
                filtered_search_times.append(end_time - start_time)
            
            # ID retrieval benchmark
            print("\n4. Milvus ID Retrieval Performance")
            id_retrieval_times = []
            for _ in tqdm(range(iterations), desc="Milvus ID retrievals"):
                # Get random IDs from the collection
                random_ids = [np.random.randint(1, 100000) for _ in range(10)]
                start_time = time.time()
                
                # Retrieve by ID in Milvus
                results = collection.query(
                    expr=f"id in {random_ids}",
                    output_fields=["id", "text_content", "metadata", "vector"]
                )
                
                end_time = time.time()
                id_retrieval_times.append(end_time - start_time)
            
            # Concurrent search benchmark
            print(f"\n5. Milvus Concurrent Search Performance ({concurrent_queries} queries, 10 workers)")
            def milvus_search_worker():
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                results = collection.search(
                    data=[query_vector],
                    anns_field="vector",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=10
                )
                
                return time.time() - start_time
            
            concurrent_search_times = []
            concurrent_queries = max(10, iterations)  # At least 10 queries for meaningful concurrent testing
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(milvus_search_worker) for _ in range(concurrent_queries)]
                concurrent_search_times = [future.result() for future in futures]
            
            # Single insert benchmark
            print("\n6. Milvus Single Insert Performance")
            single_insert_times = []
            # Use timestamp-based ID to avoid conflicts
            base_id = int(time.time() * 1000)  # Use milliseconds since epoch
            for i in tqdm(range(iterations), desc="Milvus single inserts"):
                vector = self.generate_query_vector()
                unique_id = base_id + i  # Use timestamp-based ID
                
                start_time = time.time()
                
                # Insert into Milvus
                data = [
                    [vector],  # vector field
                    [f"Test document {unique_id}"],  # text_content field
                    [f'{{"source": "test", "id": {unique_id}}}']  # metadata field
                ]
                collection.insert(data)
                collection.flush()
                
                end_time = time.time()
                single_insert_times.append(end_time - start_time)
            
            # Batch insert benchmark
            print("\n7. Milvus Batch Insert Performance")
            batch_insert_times = []
            base_id = int(time.time() * 1000) + 100000  # Use different base ID
            batch_iterations = max(1, iterations // 10)
            for i in tqdm(range(batch_iterations), desc="Milvus batch inserts"):
                start_time = time.time()
                
                # Insert 100 records in batch
                vectors = [self.generate_query_vector() for _ in range(100)]
                texts = [f"Batch document {base_id + (i * 100) + j}" for j in range(100)]
                metadata = [f'{{"source": "batch", "id": {base_id + (i * 100) + j}}}' for j in range(100)]
                
                data = [vectors, texts, metadata]
                collection.insert(data)
                collection.flush()
                
                end_time = time.time()
                batch_insert_times.append(end_time - start_time)
            
            # Update benchmark
            print("\n8. Milvus Update Performance")
            update_times = []
            for i in tqdm(range(iterations), desc="Milvus updates"):
                # Get a random existing record
                random_id = np.random.randint(1, 100000)
                new_vector = self.generate_query_vector()
                
                start_time = time.time()
                
                # Update in Milvus (delete and insert)
                collection.delete(f"id == {random_id}")
                collection.flush()
                
                data = [
                    [new_vector],
                    [f"Updated document {random_id}"],
                    [f'{{"source": "updated", "id": {random_id}}}']
                ]
                collection.insert(data)
                collection.flush()
                
                end_time = time.time()
                update_times.append(end_time - start_time)
            
            # Delete benchmark
            print("\n9. Milvus Delete Performance")
            delete_times = []
            for i in tqdm(range(iterations), desc="Milvus deletes"):
                # Get a random existing record
                random_id = np.random.randint(1, 100000)
                
                start_time = time.time()
                
                # Delete from Milvus
                collection.delete(f"id == {random_id}")
                collection.flush()
                
                end_time = time.time()
                delete_times.append(end_time - start_time)
            
            connections.disconnect(connection_alias)
            
            return {
                "single_search": {
                    "mean": statistics.mean(single_search_times),
                    "qps": 1.0 / statistics.mean(single_search_times)
                },
                "batch_search": {
                    "mean": statistics.mean(batch_search_times),
                    "qps": (10 * len(batch_search_times)) / sum(batch_search_times)
                },
                "filtered_search": {
                    "mean": statistics.mean(filtered_search_times),
                    "qps": 1.0 / statistics.mean(filtered_search_times)
                },
                "retrieve_by_id": {
                    "mean": statistics.mean(id_retrieval_times),
                    "qps": 1.0 / statistics.mean(id_retrieval_times)
                },
                "concurrent_search": {
                    "mean": statistics.mean(concurrent_search_times),
                    "qps": 1.0 / statistics.mean(concurrent_search_times)
                },
                "single_insert": {
                    "mean": statistics.mean(single_insert_times),
                    "throughput": 1.0 / statistics.mean(single_insert_times)
                },
                "batch_insert_100": {
                    "mean": statistics.mean(batch_insert_times),
                    "throughput": (100 * len(batch_insert_times)) / sum(batch_insert_times)
                },
                "update": {
                    "mean": statistics.mean(update_times),
                    "throughput": 1.0 / statistics.mean(update_times)
                },
                "delete": {
                    "mean": statistics.mean(delete_times),
                    "throughput": 1.0 / statistics.mean(delete_times)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_weaviate_benchmark(self, class_name: str = "TestVectors", iterations: int = 100):
        """Run Weaviate benchmark (read and write operations)"""
        if not WEAVIATE_AVAILABLE:
            return {"error": "weaviate-client not available. Install with: pip install weaviate-client"}
        
        try:
            # Connect to Weaviate using v4 client
            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port
            )
            
            if not client.is_ready():
                return {"error": "Weaviate server not ready"}
            
            if not client.collections.exists(class_name):
                return {"error": f"Class '{class_name}' does not exist"}
            
            # Get the collection
            collection = client.collections.get(class_name)
            
            # Single search benchmark
            print("\n1. Weaviate Single Search Performance")
            single_search_times = []
            for _ in tqdm(range(iterations), desc="Weaviate single searches"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                # Search in Weaviate using v4 API
                result = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=10,
                    return_metadata=["distance"]
                )
                
                end_time = time.time()
                single_search_times.append(end_time - start_time)
            
            # Batch search benchmark
            print("\n2. Weaviate Batch Search Performance")
            batch_search_times = []
            batch_iterations = max(1, iterations // 10)
            for _ in tqdm(range(batch_iterations), desc="Weaviate batch searches"):
                query_vectors = [self.generate_query_vector() for _ in range(10)]
                start_time = time.time()
                
                # Batch search in Weaviate
                for query_vector in query_vectors:
                    result = collection.query.near_vector(
                        near_vector=query_vector,
                        limit=10,
                        return_metadata=["distance"]
                    )
                
                end_time = time.time()
                batch_search_times.append(end_time - start_time)
            
            # Filtered search benchmark
            print("\n3. Weaviate Filtered Search Performance")
            filtered_search_times = []
            for _ in tqdm(range(iterations), desc="Weaviate filtered searches"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                # Search with filter in Weaviate (simplified for now)
                result = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=10,
                    return_metadata=["distance"]
                )
                
                end_time = time.time()
                filtered_search_times.append(end_time - start_time)
            
            # ID retrieval benchmark
            print("\n4. Weaviate ID Retrieval Performance")
            id_retrieval_times = []
            for _ in tqdm(range(iterations), desc="Weaviate ID retrievals"):
                # Get random IDs from the collection
                random_ids = [f"doc_{np.random.randint(1, 100000)}" for _ in range(10)]
                start_time = time.time()
                
                # Retrieve by ID in Weaviate
                for doc_id in random_ids:
                    try:
                        result = collection.query.fetch_object_by_id(doc_id)
                    except:
                        pass  # Ignore if ID doesn't exist
                
                end_time = time.time()
                id_retrieval_times.append(end_time - start_time)
            
            # Concurrent search benchmark
            print(f"\n5. Weaviate Concurrent Search Performance ({concurrent_queries} queries, 10 workers)")
            def weaviate_search_worker():
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                result = collection.query.near_vector(
                    near_vector=query_vector,
                    limit=10,
                    return_metadata=["distance"]
                )
                
                return time.time() - start_time
            
            concurrent_search_times = []
            concurrent_queries = max(10, iterations)  # At least 10 queries for meaningful concurrent testing
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(weaviate_search_worker) for _ in range(concurrent_queries)]
                concurrent_search_times = [future.result() for future in futures]
            
            # Single insert benchmark
            print("\n6. Weaviate Single Insert Performance")
            single_insert_times = []
            # Use timestamp-based ID to avoid conflicts
            base_id = int(time.time() * 1000)  # Use milliseconds since epoch
            for i in tqdm(range(iterations), desc="Weaviate single inserts"):
                vector = self.generate_query_vector()
                unique_id = f"doc_{base_id + i}"
                
                start_time = time.time()
                
                # Insert into Weaviate using v4 API
                data_object = {
                    "text_content": f"Test document {unique_id}",
                    "metadata": f'{{"source": "test", "id": "{unique_id}"}}'
                }
                
                collection.data.insert(
                    properties=data_object,
                    vector=vector
                )
                
                end_time = time.time()
                single_insert_times.append(end_time - start_time)
            
            # Batch insert benchmark
            print("\n7. Weaviate Batch Insert Performance")
            batch_insert_times = []
            base_id = int(time.time() * 1000) + 100000  # Use different base ID
            batch_iterations = max(1, iterations // 10)
            for i in tqdm(range(batch_iterations), desc="Weaviate batch inserts"):
                start_time = time.time()
                
                # Insert 100 records in batch
                data_objects = []
                for j in range(100):
                    vector = self.generate_query_vector()
                    unique_id = f"doc_{base_id + (i * 100) + j}"
                    data_objects.append(wvc.data.DataObject(
                        properties={
                            "text_content": f"Batch document {unique_id}",
                            "metadata": f'{{"source": "batch", "id": "{unique_id}"}}'
                        },
                        vector=vector
                    ))
                
                collection.data.insert_many(data_objects)
                
                end_time = time.time()
                batch_insert_times.append(end_time - start_time)
            
            # Update benchmark
            print("\n8. Weaviate Update Performance")
            update_times = []
            for i in tqdm(range(iterations), desc="Weaviate updates"):
                # Get a random existing record
                random_id = f"doc_{np.random.randint(1, 100000)}"
                new_vector = self.generate_query_vector()
                
                start_time = time.time()
                
                # Update in Weaviate (delete and insert)
                try:
                    collection.data.delete_by_id(random_id)
                except:
                    pass  # Ignore if ID doesn't exist
                
                data_object = {
                    "text_content": f"Updated document {random_id}",
                    "metadata": f'{{"source": "updated", "id": "{random_id}"}}'
                }
                
                collection.data.insert(
                    properties=data_object,
                    vector=new_vector
                )
                
                end_time = time.time()
                update_times.append(end_time - start_time)
            
            # Delete benchmark
            print("\n9. Weaviate Delete Performance")
            delete_times = []
            for i in tqdm(range(iterations), desc="Weaviate deletes"):
                # Get a random existing record
                random_id = f"doc_{np.random.randint(1, 100000)}"
                
                start_time = time.time()
                
                # Delete from Weaviate
                try:
                    collection.data.delete_by_id(random_id)
                except:
                    pass  # Ignore if ID doesn't exist
                
                end_time = time.time()
                delete_times.append(end_time - start_time)
            
            client.close()
            
            return {
                "single_search": {
                    "mean": statistics.mean(single_search_times),
                    "qps": 1.0 / statistics.mean(single_search_times)
                },
                "batch_search": {
                    "mean": statistics.mean(batch_search_times),
                    "qps": (10 * len(batch_search_times)) / sum(batch_search_times)
                },
                "filtered_search": {
                    "mean": statistics.mean(filtered_search_times),
                    "qps": 1.0 / statistics.mean(filtered_search_times)
                },
                "retrieve_by_id": {
                    "mean": statistics.mean(id_retrieval_times),
                    "qps": 1.0 / statistics.mean(id_retrieval_times)
                },
                "concurrent_search": {
                    "mean": statistics.mean(concurrent_search_times),
                    "qps": 1.0 / statistics.mean(concurrent_search_times)
                },
                "single_insert": {
                    "mean": statistics.mean(single_insert_times),
                    "throughput": 1.0 / statistics.mean(single_insert_times)
                },
                "batch_insert_100": {
                    "mean": statistics.mean(batch_insert_times),
                    "throughput": (100 * len(batch_insert_times)) / sum(batch_insert_times)
                },
                "update": {
                    "mean": statistics.mean(update_times),
                    "throughput": 1.0 / statistics.mean(update_times)
                },
                "delete": {
                    "mean": statistics.mean(delete_times),
                    "throughput": 1.0 / statistics.mean(delete_times)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    
    # ==================== MAIN BENCHMARK RUNNER ====================
    
    def run_comprehensive_benchmark(self, 
                                  read_collection: str = "test_vectors",
                                  write_collection: str = "test_vectors",
                                  iterations: int = 100,
                                  load_duration: int = 120,
                                  run_read: bool = True,
                                  run_write: bool = True,
                                  run_postgres: bool = True,
                                  run_postgres_ts: bool = True,
                                  run_comparison: bool = True,
                                  run_load_test: bool = True,
                                  run_milvus: bool = False,
                                  run_weaviate: bool = False):
        """Run comprehensive benchmark suite based on flags"""
        
        print("Comprehensive Vector Database Benchmark Suite")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Testing Databases: Weaviate, Qdrant, Milvus, TimescaleDB, PostgreSQL")
        print(f"Qdrant Host: {self.qdrant_host}:{self.qdrant_port}")
        print(f"PostgreSQL Host: {self.postgres_config['host']}:{self.postgres_config['port']}")
        print(f"TimescaleDB Host: {self.postgres_ts_config['host']}:{self.postgres_ts_config['port']}")
        print(f"Milvus Host: {self.milvus_host}:{self.milvus_port}")
        print(f"Weaviate Host: {self.weaviate_host}:{self.weaviate_port}")
        print(f"Read Collection: {read_collection}")
        print(f"Write Collection: {write_collection}")
        print(f"Iterations: {iterations}")
        print(f"Load Test Duration: {load_duration}s")
        print(f"Vector Dimension: {self.vector_dim}")
        
        # Get system info
        system_info = self.get_system_info()
        print(f"\nSystem Info:")
        print(f"CPU Cores: {system_info['cpu_count']}")
        print(f"Memory: {system_info['memory_total'] / (1024**3):.2f} GB")
        print(f"Available Memory: {system_info['memory_available'] / (1024**3):.2f} GB")
        print(f"Disk Usage: {system_info['disk_usage']:.1f}%")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "iterations": iterations,
                "load_duration": load_duration,
                "vector_dimension": self.vector_dim,
                "system_info": system_info,
                "databases_tested": ["Weaviate", "Qdrant", "Milvus", "TimescaleDB", "PostgreSQL"]
            },
            "weaviate_benchmark": None,
            "read_benchmark": None,
            "write_benchmark": None,
            "milvus_benchmark": None,
            "postgres_ts_benchmark": None,
            "postgres_benchmark": None,
            "database_comparison": None,
            "load_test": None
        }
        
        try:
            # Database execution order: Weaviate, Qdrant, Milvus, TimescaleDB, PostgreSQL
            
            # 1. Run Weaviate benchmark
            if run_weaviate:
                print("\n" + "="*60)
                print("RUNNING WEAVIATE BENCHMARK (1/5)")
                print("="*60)
                results["weaviate_benchmark"] = self.run_weaviate_benchmark("TestVectors", iterations)
            
            # 2. Run Qdrant benchmarks (read and write)
            if run_read:
                print("\n" + "="*60)
                print("RUNNING QDRANT READ BENCHMARK (2/5)")
                print("="*60)
                results["read_benchmark"] = self.run_read_benchmark(read_collection, iterations)
            
            if run_write:
                print("\n" + "="*60)
                print("RUNNING QDRANT WRITE BENCHMARK (2/5)")
                print("="*60)
                # Don't cleanup if using the same collection as read benchmark
                cleanup_write = (write_collection != read_collection)
                results["write_benchmark"] = self.run_write_benchmark(write_collection, iterations, cleanup=cleanup_write)
            
            # 3. Run Milvus benchmark
            if run_milvus:
                print("\n" + "="*60)
                print("RUNNING MILVUS BENCHMARK (3/5)")
                print("="*60)
                results["milvus_benchmark"] = self.run_milvus_benchmark(read_collection, iterations)
            
            # 4. Run TimescaleDB benchmark
            if run_postgres_ts:
                print("\n" + "="*60)
                print("RUNNING TIMESCALEDB BENCHMARK (4/5)")
                print("="*60)
                results["postgres_ts_benchmark"] = self.run_postgres_ts_benchmark(iterations)
            
            # 5. Run PostgreSQL benchmark
            if run_postgres:
                print("\n" + "="*60)
                print("RUNNING POSTGRESQL BENCHMARK (5/5)")
                print("="*60)
                results["postgres_benchmark"] = self.run_postgres_benchmark(iterations)
            
            # Run database comparison (after all individual benchmarks)
            if run_comparison:
                try:
                    print("\n" + "="*60)
                    print("RUNNING DATABASE COMPARISON")
                    print("="*60)
                    results["database_comparison"] = self.run_database_comparison(read_collection, iterations)
                except Exception as e:
                    print(f"❌ Database comparison failed: {e}")
                    results["database_comparison"] = {"error": str(e)}
            
            # Run load test (final test)
            if run_load_test:
                print("\n" + "="*60)
                print("RUNNING LOAD TEST")
                print("="*60)
                results["load_test"] = self.run_load_test(read_collection, write_collection, load_duration)
            
            # Print summary
            self.print_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Error during comprehensive benchmark: {e}")
            raise
    
    def print_summary(self, results):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        if results["read_benchmark"]:
            print("\nREAD PERFORMANCE:")
            read_results = results["read_benchmark"]
            for operation, stats in read_results.items():
                print(f"  {operation}: {stats['mean']:.4f}s mean, {stats['qps']:.2f} QPS")
        
        if results["write_benchmark"]:
            print("\nWRITE PERFORMANCE:")
            write_results = results["write_benchmark"]
            for operation, stats in write_results.items():
                if 'batch_size' in stats:
                    print(f"  {operation}: {stats['mean']:.4f}s mean, {stats['throughput']:.2f} points/sec")
                else:
                    print(f"  {operation}: {stats['mean']:.4f}s mean, {stats['throughput']:.2f} ops/sec")
        
        if results["postgres_benchmark"]:
            print("\nPOSTGRESQL PERFORMANCE:")
            postgres_results = results["postgres_benchmark"]
            for operation, stats in postgres_results.items():
                if 'qps' in stats:
                    print(f"  {operation}: {stats['mean']:.4f}s mean, {stats['qps']:.2f} QPS")
                else:
                    print(f"  {operation}: {stats['mean']:.4f}s mean, {stats['throughput']:.2f} ops/sec")
        
        if results["database_comparison"] and "ratios" in results["database_comparison"]:
            print("\nDATABASE COMPARISON:")
            ratios = results["database_comparison"]["ratios"]
            for metric, ratio in ratios.items():
                if 'search' in metric:
                    print(f"  Search Performance: Qdrant is {ratio['qdrant_vs_postgres']:.1f}x {'faster' if ratio['qdrant_vs_postgres'] > 1 else 'slower'}")
                elif 'insert' in metric:
                    print(f"  Insert Performance: Qdrant is {ratio['qdrant_vs_postgres']:.1f}x {'faster' if ratio['qdrant_vs_postgres'] > 1 else 'slower'}")
        
        if results["load_test"]:
            print("\nLOAD TEST RESULTS:")
            load_stats = results["load_test"]
            print(f"  CPU Usage: {load_stats['cpu_usage']['mean']:.1f}% mean, {load_stats['cpu_usage']['max']:.1f}% max")
            print(f"  Memory Usage: {load_stats['memory_usage']['mean']:.1f}% mean, {load_stats['memory_usage']['max']:.1f}% max")
        
        # New database performance summaries
        if results["milvus_benchmark"]:
            print("\nMILVUS PERFORMANCE:")
            milvus_results = results["milvus_benchmark"]
            if "error" in milvus_results:
                print(f"  Error: {milvus_results['error']}")
            else:
                print(f"  Search: {milvus_results['single_search']['mean']:.4f}s mean, {milvus_results['single_search']['qps']:.2f} QPS")
                print(f"  Insert: {milvus_results['single_insert']['mean']:.4f}s mean, {milvus_results['single_insert']['throughput']:.2f} ops/sec")
        
        if results["weaviate_benchmark"]:
            print("\nWEAVIATE PERFORMANCE:")
            weaviate_results = results["weaviate_benchmark"]
            if "error" in weaviate_results:
                print(f"  Error: {weaviate_results['error']}")
            else:
                print(f"  Search: {weaviate_results['single_search']['mean']:.4f}s mean, {weaviate_results['single_search']['qps']:.2f} QPS")
                print(f"  Insert: {weaviate_results['single_insert']['mean']:.4f}s mean, {weaviate_results['single_insert']['throughput']:.2f} ops/sec")
        
        # Add comprehensive performance comparison summary
        self.print_performance_comparison_summary(results)
        
        # Print summary table
        self.print_summary_table(results)
    
    def print_performance_comparison_summary(self, results):
        """Print comprehensive performance comparison across all databases"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATABASE PERFORMANCE COMPARISON")
        print("="*80)
        
        # Extract performance data from all databases
        qdrant_read = results.get("read_benchmark", {})
        qdrant_write = results.get("write_benchmark", {})
        postgres = results.get("postgres_benchmark", {})
        milvus = results.get("milvus_benchmark", {})
        weaviate = results.get("weaviate_benchmark", {})
        
        # Check if we have at least some data
        databases_with_data = []
        if qdrant_read:
            databases_with_data.append("Qdrant")
        if postgres:
            databases_with_data.append("PostgreSQL")
        if milvus and "error" not in milvus:
            databases_with_data.append("Milvus")
        if weaviate and "error" not in weaviate:
            databases_with_data.append("Weaviate")
        
        if len(databases_with_data) < 2:
            print("Insufficient data for performance comparison (need at least 2 databases)")
            return
        
        print("\n🔍 SEARCH PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        # Collect search performance data from all databases
        search_data = []
        
        # Qdrant search data
        if qdrant_read and "single_search" in qdrant_read:
            search_data.append({
                "name": "Qdrant",
                "qps": qdrant_read["single_search"]["qps"],
                "mean_time": qdrant_read["single_search"]["mean"],
                "type": "Vector DB"
            })
        
        # PostgreSQL search data
        if postgres and "single_search" in postgres:
            search_data.append({
                "name": "PostgreSQL",
                "qps": postgres["single_search"]["qps"],
                "mean_time": postgres["single_search"]["mean"],
                "type": "SQL + Vector"
            })
        
        # TimescaleDB search data
        if results.get("postgres_ts_benchmark") and "single_search" in results["postgres_ts_benchmark"]:
            postgres_ts = results["postgres_ts_benchmark"]
            search_data.append({
                "name": "TimescaleDB",
                "qps": postgres_ts["single_search"]["qps"],
                "mean_time": postgres_ts["single_search"]["mean_time"],
                "type": "Time-Series + Vector"
            })
        
        # Milvus search data
        if milvus and "error" not in milvus and "single_search" in milvus:
            search_data.append({
                "name": "Milvus",
                "qps": milvus["single_search"]["qps"],
                "mean_time": milvus["single_search"]["mean"],
                "type": "Vector DB"
            })
        
        # Weaviate search data
        if weaviate and "error" not in weaviate and "single_search" in weaviate:
            search_data.append({
                "name": "Weaviate",
                "qps": weaviate["single_search"]["qps"],
                "mean_time": weaviate["single_search"]["mean"],
                "type": "Vector DB"
            })
        
        if search_data:
            # Sort by QPS (descending)
            search_data.sort(key=lambda x: x["qps"], reverse=True)
            
            print("Single Vector Search Performance:")
            for i, db in enumerate(search_data):
                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"  {rank_emoji} {db['name']:12} {db['qps']:6.1f} QPS ({db['mean_time']:.4f}s) - {db['type']}")
            
            # Show performance ratios
            if len(search_data) >= 2:
                fastest = search_data[0]
                print(f"\n  🏆 {fastest['name']} is {fastest['qps']/search_data[-1]['qps']:.1f}x faster than {search_data[-1]['name']}")
        
        # Additional Qdrant-specific features
        if qdrant_read:
            print(f"\nQdrant Advanced Features:")
            if "batch_search" in qdrant_read:
                print(f"  • Batch Search (10 vectors): {qdrant_read['batch_search']['qps']:.1f} QPS")
            if "filtered_search" in qdrant_read:
                print(f"  • Filtered Search: {qdrant_read['filtered_search']['qps']:.1f} QPS")
            if "retrieve_by_id" in qdrant_read:
                print(f"  • ID Retrieval: {qdrant_read['retrieve_by_id']['qps']:.1f} QPS")
            if "concurrent_search" in qdrant_read:
                print(f"  • Concurrent Search: {qdrant_read['concurrent_search']['qps']:.1f} QPS")
        
        print("\n✏️  WRITE PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        # Collect write performance data from all databases
        write_data = []
        
        # Qdrant write data
        if qdrant_write and "single_insert" in qdrant_write:
            write_data.append({
                "name": "Qdrant",
                "ops_per_sec": qdrant_write["single_insert"]["throughput"],
                "mean_time": qdrant_write["single_insert"]["mean"],
                "type": "Vector DB"
            })
        
        # PostgreSQL write data
        if postgres and "single_insert" in postgres:
            write_data.append({
                "name": "PostgreSQL",
                "ops_per_sec": postgres["single_insert"]["throughput"],
                "mean_time": postgres["single_insert"]["mean"],
                "type": "SQL + Vector"
            })
        
        # TimescaleDB write data
        if results.get("postgres_ts_benchmark") and "single_insert" in results["postgres_ts_benchmark"]:
            postgres_ts = results["postgres_ts_benchmark"]
            write_data.append({
                "name": "TimescaleDB",
                "ops_per_sec": postgres_ts["single_insert"]["throughput"],
                "mean_time": postgres_ts["single_insert"]["mean_time"],
                "type": "Time-Series + Vector"
            })
        
        # Milvus write data
        if milvus and "error" not in milvus and "single_insert" in milvus:
            write_data.append({
                "name": "Milvus",
                "ops_per_sec": milvus["single_insert"]["throughput"],
                "mean_time": milvus["single_insert"]["mean"],
                "type": "Vector DB"
            })
        
        # Weaviate write data
        if weaviate and "error" not in weaviate and "single_insert" in weaviate:
            write_data.append({
                "name": "Weaviate",
                "ops_per_sec": weaviate["single_insert"]["throughput"],
                "mean_time": weaviate["single_insert"]["mean"],
                "type": "Vector DB"
            })
        
        if write_data:
            # Sort by ops/sec (descending)
            write_data.sort(key=lambda x: x["ops_per_sec"], reverse=True)
            
            print("Single Insert Performance:")
            for i, db in enumerate(write_data):
                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"  {rank_emoji} {db['name']:12} {db['ops_per_sec']:6.1f} ops/sec ({db['mean_time']:.4f}s) - {db['type']}")
            
            # Show performance ratios
            if len(write_data) >= 2:
                fastest = write_data[0]
                print(f"\n  🏆 {fastest['name']} is {fastest['ops_per_sec']/write_data[-1]['ops_per_sec']:.1f}x faster than {write_data[-1]['name']}")
        
        # Additional Qdrant-specific write features
        if qdrant_write:
            print(f"\nQdrant Advanced Write Features:")
            if "batch_insert_10" in qdrant_write:
                print(f"  • Batch Insert (10 points): {qdrant_write['batch_insert_10']['throughput']:.1f} points/sec")
            if "batch_insert_100" in qdrant_write:
                print(f"  • Batch Insert (100 points): {qdrant_write['batch_insert_100']['throughput']:.1f} points/sec")
            if "batch_insert_1000" in qdrant_write:
                print(f"  • Batch Insert (1000 points): {qdrant_write['batch_insert_1000']['throughput']:.1f} points/sec")
            if "update" in qdrant_write:
                print(f"  • Update Operations: {qdrant_write['update']['throughput']:.1f} ops/sec")
        if "delete" in qdrant_write:
                print(f"  • Delete Operations: {qdrant_write['delete']['throughput']:.1f} ops/sec")
        
        print("\n📊 OVERALL PERFORMANCE INSIGHTS:")
        print("-" * 50)
        
        # Database performance summary
        if search_data and write_data:
            print("Database Performance Summary:")
            for db in search_data:
                write_perf = next((w for w in write_data if w["name"] == db["name"]), None)
                if write_perf:
                    print(f"  • {db['name']}: {db['qps']:.1f} QPS search, {write_perf['ops_per_sec']:.1f} ops/sec write")
                else:
                    print(f"  • {db['name']}: {db['qps']:.1f} QPS search, N/A write")
        
        # Performance rankings
        if search_data:
            fastest_search = search_data[0]
            print(f"\n🏆 Search Performance Winner: {fastest_search['name']} ({fastest_search['qps']:.1f} QPS)")
        
        if write_data:
            fastest_write = write_data[0]
            print(f"🏆 Write Performance Winner: {fastest_write['name']} ({fastest_write['ops_per_sec']:.1f} ops/sec)")
        
        # Additional insights
        if qdrant_read:
            if "concurrent_search" in qdrant_read:
                concurrent_qps = qdrant_read["concurrent_search"]["qps"]
                print(f"\n• Qdrant Concurrent Search: {concurrent_qps:.1f} QPS under load")
            
            if "scroll" in qdrant_read:
                scroll_qps = qdrant_read["scroll"]["qps"]
                print(f"• Qdrant Large Dataset Scrolling: {scroll_qps:.1f} QPS for bulk operations")
        
        # Memory and CPU insights
        if results.get("load_test"):
            load_stats = results["load_test"]
            print(f"\n• System Load: {load_stats['cpu_usage']['mean']:.1f}% CPU, {load_stats['memory_usage']['mean']:.1f}% Memory during sustained load")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        # Database-specific recommendations based on performance
        if search_data and write_data:
            print("Choose your database based on performance and requirements:")
            print()
            
            # Qdrant recommendations
            if any(db["name"] == "Qdrant" for db in search_data):
                print("🚀 QDRANT - Best for high-performance vector operations:")
                print("  ✅ Fastest search performance in most cases")
                print("  ✅ Excellent batch operations and concurrent processing")
                print("  ✅ Advanced features: filtering, scrolling, ID retrieval")
                print("  ✅ Microservices architectures")
                print("  ✅ Real-time applications")
                print("  ❌ No SQL integration")
                print()
            
            # PostgreSQL recommendations
            if any(db["name"] == "PostgreSQL" for db in search_data):
                print("🐘 POSTGRESQL - Best for SQL integration:")
                print("  ✅ Full SQL compatibility with vector search")
                print("  ✅ ACID compliance and transactions")
                print("  ✅ Complex relational queries with vectors")
                print("  ✅ Existing PostgreSQL ecosystems")
                print("  ❌ Slower vector operations than dedicated vector DBs")
                print()
            
            # TimescaleDB recommendations
            if any(db["name"] == "TimescaleDB" for db in search_data):
                print("⏰ TIMESCALEDB - Best for time-series + vector data:")
                print("  ✅ Combines time-series and vector search capabilities")
                print("  ✅ Excellent for temporal vector data")
                print("  ✅ Built on PostgreSQL foundation")
                print("  ✅ Automatic partitioning and optimization")
                print("  ✅ Time-based analytics with vector similarity")
                print("  ❌ More complex setup than standard PostgreSQL")
                print()
            
            # Milvus recommendations
            if any(db["name"] == "Milvus" for db in search_data):
                milvus_search = next((db for db in search_data if db["name"] == "Milvus"), None)
                milvus_write = next((db for db in write_data if db["name"] == "Milvus"), None)
                print("🚀 MILVUS - Best for large-scale vector operations:")
                print("  ✅ Good search performance")
                print("  ✅ Designed for massive scale")
                print("  ✅ Distributed architecture")
                print("  ❌ Slower write operations")
                if milvus_write and milvus_write["ops_per_sec"] < 1:
                    print("  ⚠️  Very slow writes - consider for read-heavy workloads")
                print()
            
            # Weaviate recommendations
            if any(db["name"] == "Weaviate" for db in search_data):
                print("🔮 WEAVIATE - Best for AI/ML integration:")
                print("  ✅ Good balance of search and write performance")
                print("  ✅ Built-in vectorization capabilities")
                print("  ✅ GraphQL API")
                print("  ✅ AI/ML ecosystem integration")
                print("  ❌ Less advanced vector operations than Qdrant")
                print()
        
        # General recommendations
        print("🎯 GENERAL RECOMMENDATIONS:")
        print("  • For maximum vector performance: Qdrant")
        print("  • For SQL integration: PostgreSQL + pgvector")
        print("  • For time-series + vectors: TimescaleDB")
        print("  • For large scale deployments: Milvus")
        print("  • For AI/ML workflows: Weaviate")
        print("  • For production: Consider your specific use case, scale, and integration requirements")
    
    def print_summary_table(self, results):
        """Print a comprehensive summary table of all database performance metrics"""
        print("\n" + "="*120)
        print("COMPREHENSIVE PERFORMANCE SUMMARY TABLE")
        print("="*120)
        
        # Extract performance data from all databases
        qdrant_read = results.get("read_benchmark", {})
        qdrant_write = results.get("write_benchmark", {})
        postgres = results.get("postgres_benchmark", {})
        postgres_ts = results.get("postgres_ts_benchmark", {})
        milvus = results.get("milvus_benchmark", {})
        weaviate = results.get("weaviate_benchmark", {})
        
        # Prepare data for table
        table_data = []
        
        # Qdrant data
        if qdrant_read and qdrant_write:
            table_data.append({
                "Database": "Qdrant",
                "Type": "Vector DB",
                "Search QPS": f"{round(qdrant_read.get('single_search', {}).get('qps', 0))}",
                "Batch Search QPS": f"{round(qdrant_read.get('batch_search', {}).get('qps', 0))}",
                "Filtered Search QPS": f"{round(qdrant_read.get('filtered_search', {}).get('qps', 0))}",
                "ID Retrieval QPS": f"{round(qdrant_read.get('retrieve_by_id', {}).get('qps', 0))}",
                "Concurrent Search QPS": f"{round(qdrant_read.get('concurrent_search', {}).get('qps', 0))}",
                "Single Insert ops/sec": f"{round(qdrant_write.get('single_insert', {}).get('throughput', 0))}",
                "Batch Insert (100) ops/sec": f"{round(qdrant_write.get('batch_insert_100', {}).get('throughput', 0))}",
                "Update ops/sec": f"{round(qdrant_write.get('update', {}).get('throughput', 0))}",
                "Delete ops/sec": f"{round(qdrant_write.get('delete', {}).get('throughput', 0))}"
            })
        
        # PostgreSQL data
        if postgres:
            table_data.append({
                "Database": "PostgreSQL",
                "Type": "SQL + Vector",
                "Search QPS": f"{round(postgres.get('single_search', {}).get('qps', 0))}",
                "Batch Search QPS": f"{round(postgres.get('batch_search', {}).get('qps', 0))}",
                "Filtered Search QPS": f"{round(postgres.get('filtered_search', {}).get('qps', 0))}",
                "ID Retrieval QPS": f"{round(postgres.get('retrieve_by_id', {}).get('qps', 0))}",
                "Concurrent Search QPS": f"{round(postgres.get('concurrent_search', {}).get('qps', 0))}",
                "Single Insert ops/sec": f"{round(postgres.get('single_insert', {}).get('throughput', 0))}",
                "Batch Insert (100) ops/sec": f"{round(postgres.get('batch_insert_100', {}).get('throughput', 0))}",
                "Update ops/sec": f"{round(postgres.get('update', {}).get('throughput', 0))}",
                "Delete ops/sec": f"{round(postgres.get('delete', {}).get('throughput', 0))}"
            })
        
        # TimescaleDB data
        if postgres_ts and "error" not in postgres_ts:
            table_data.append({
                "Database": "TimescaleDB",
                "Type": "Time-Series + Vector",
                "Search QPS": f"{round(postgres_ts.get('single_search', {}).get('qps', 0))}",
                "Batch Search QPS": f"{round(postgres_ts.get('batch_search', {}).get('qps', 0))}",
                "Filtered Search QPS": f"{round(postgres_ts.get('filtered_search', {}).get('qps', 0))}",
                "ID Retrieval QPS": f"{round(postgres_ts.get('retrieve_by_id', {}).get('qps', 0))}",
                "Concurrent Search QPS": f"{round(postgres_ts.get('concurrent_search', {}).get('qps', 0))}",
                "Single Insert ops/sec": f"{round(postgres_ts.get('single_insert', {}).get('throughput', 0))}",
                "Batch Insert (100) ops/sec": f"{round(postgres_ts.get('batch_insert_100', {}).get('throughput', 0))}",
                "Update ops/sec": f"{round(postgres_ts.get('update', {}).get('throughput', 0))}",
                "Delete ops/sec": f"{round(postgres_ts.get('delete', {}).get('throughput', 0))}"
            })
        
        # Milvus data
        if milvus and "error" not in milvus:
            table_data.append({
                "Database": "Milvus",
                "Type": "Vector DB",
                "Search QPS": f"{round(milvus.get('single_search', {}).get('qps', 0))}",
                "Batch Search QPS": f"{round(milvus.get('batch_search', {}).get('qps', 0))}",
                "Filtered Search QPS": f"{round(milvus.get('filtered_search', {}).get('qps', 0))}",
                "ID Retrieval QPS": f"{round(milvus.get('retrieve_by_id', {}).get('qps', 0))}",
                "Concurrent Search QPS": f"{round(milvus.get('concurrent_search', {}).get('qps', 0))}",
                "Single Insert ops/sec": f"{round(milvus.get('single_insert', {}).get('throughput', 0))}",
                "Batch Insert (100) ops/sec": f"{round(milvus.get('batch_insert_100', {}).get('throughput', 0))}",
                "Update ops/sec": f"{round(milvus.get('update', {}).get('throughput', 0))}",
                "Delete ops/sec": f"{round(milvus.get('delete', {}).get('throughput', 0))}"
            })
        
        # Weaviate data
        if weaviate and "error" not in weaviate:
            table_data.append({
                "Database": "Weaviate",
                "Type": "Vector DB",
                "Search QPS": f"{round(weaviate.get('single_search', {}).get('qps', 0))}",
                "Batch Search QPS": f"{round(weaviate.get('batch_search', {}).get('qps', 0))}",
                "Filtered Search QPS": f"{round(weaviate.get('filtered_search', {}).get('qps', 0))}",
                "ID Retrieval QPS": f"{round(weaviate.get('retrieve_by_id', {}).get('qps', 0))}",
                "Concurrent Search QPS": f"{round(weaviate.get('concurrent_search', {}).get('qps', 0))}",
                "Single Insert ops/sec": f"{round(weaviate.get('single_insert', {}).get('throughput', 0))}",
                "Batch Insert (100) ops/sec": f"{round(weaviate.get('batch_insert_100', {}).get('throughput', 0))}",
                "Update ops/sec": f"{round(weaviate.get('update', {}).get('throughput', 0))}",
                "Delete ops/sec": f"{round(weaviate.get('delete', {}).get('throughput', 0))}"
            })
        
        if not table_data:
            print("No performance data available for summary table")
            return
        
        # Print table header
        print(f"{'Database':<12} {'Type':<12} {'Search':<8} {'Batch':<8} {'Filtered':<10} {'ID Ret.':<8} {'Concurrent':<10} {'Insert':<8} {'Batch Ins':<10} {'Update':<8} {'Delete':<8}")
        print("-" * 120)
        
        # Find best values for each metric
        metrics = ['Search QPS', 'Batch Search QPS', 'Filtered Search QPS', 'ID Retrieval QPS', 
                  'Concurrent Search QPS', 'Single Insert ops/sec', 'Batch Insert (100) ops/sec', 
                  'Update ops/sec', 'Delete ops/sec']
        
        best_values = {}
        for metric in metrics:
            best_value = 0
            for row in table_data:
                try:
                    value = int(row[metric])
                    if value > best_value:
                        best_value = value
                except (ValueError, TypeError):
                    pass
            best_values[metric] = best_value
        
        # Print table rows with bold formatting for best values
        for row in table_data:
            formatted_row = []
            column_widths = [12, 12, 8, 8, 10, 8, 10, 8, 10, 8, 8]
            metrics = ['Database', 'Type', 'Search QPS', 'Batch Search QPS', 'Filtered Search QPS', 
                      'ID Retrieval QPS', 'Concurrent Search QPS', 'Single Insert ops/sec', 
                      'Batch Insert (100) ops/sec', 'Update ops/sec', 'Delete ops/sec']
            
            for i, metric in enumerate(metrics):
                width = column_widths[i]
                if metric in ['Database', 'Type']:
                    formatted_row.append(f"{row[metric]:<{width}}")
            else:
                    try:
                        value = int(row[metric])
                        if value == best_values[metric] and value > 0:
                            formatted_row.append(f"\033[1m{row[metric]:<{width}}\033[0m")  # Bold
                        else:
                            formatted_row.append(f"{row[metric]:<{width}}")
                    except (ValueError, TypeError):
                        formatted_row.append(f"{row[metric]:<{width}}")
            
            print(" ".join(formatted_row))
        
        # Add performance rankings
        print("\n" + "="*80)
        print("PERFORMANCE RANKINGS")
        print("="*80)
        
        # Search performance ranking
        search_rankings = []
        for row in table_data:
            if row['Search QPS'] != 'N/A':
                search_rankings.append((row['Database'], float(row['Search QPS'])))
        
        if search_rankings:
            search_rankings.sort(key=lambda x: x[1], reverse=True)
            print("\n🔍 SEARCH PERFORMANCE RANKING:")
            for i, (db, qps) in enumerate(search_rankings, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                print(f"  {medal} {i}. {db}: {qps:.1f} QPS")
        
        # Write performance ranking
        write_rankings = []
        for row in table_data:
            if row['Single Insert ops/sec'] != 'N/A':
                write_rankings.append((row['Database'], float(row['Single Insert ops/sec'])))
        
        if write_rankings:
            write_rankings.sort(key=lambda x: x[1], reverse=True)
            print("\n✏️  WRITE PERFORMANCE RANKING:")
            for i, (db, ops) in enumerate(write_rankings, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                print(f"  {medal} {i}. {db}: {ops:.1f} ops/sec")
        
        # Overall performance summary
        print("\n" + "="*80)
        print("OVERALL PERFORMANCE SUMMARY")
        print("="*80)
        
        if search_rankings and write_rankings:
            # Find best performers
            best_search = search_rankings[0]
            best_write = write_rankings[0]
            
            print(f"🏆 Best Search Performance: {best_search[0]} ({best_search[1]:.1f} QPS)")
            print(f"🏆 Best Write Performance: {best_write[0]} ({best_write[1]:.1f} ops/sec)")
            
            # Calculate performance ratios
            if len(search_rankings) >= 2:
                fastest_qps = search_rankings[0][1]
                slowest_qps = search_rankings[-1][1]
                if slowest_qps > 0:
                    search_ratio = fastest_qps / slowest_qps
                    print(f"📊 Search Performance Range: {search_ratio:.1f}x difference between fastest and slowest")
                else:
                    print(f"📊 Search Performance Range: Cannot calculate ratio (slowest QPS is 0)")
            
            if len(write_rankings) >= 2:
                fastest_ops = write_rankings[0][1]
                slowest_ops = write_rankings[-1][1]
                if slowest_ops > 0:
                    write_ratio = fastest_ops / slowest_ops
                    print(f"📊 Write Performance Range: {write_ratio:.1f}x difference between fastest and slowest")
                else:
                    print(f"📊 Write Performance Range: {fastest_ops:.1f}x difference (slowest is 0)")
    
    def save_results(self, results, filename: str = "comprehensive_benchmark_results.json"):
        """Save comprehensive results to JSON file"""
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Ensure filename is in results directory
        if not filename.startswith("results/"):
            filename = f"results/{filename}"
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nComprehensive results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Vector Database Benchmark Suite")
    
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
    parser.add_argument("--read-collection", default="test_vectors", help="Read benchmark collection")
    parser.add_argument("--write-collection", default="test_vectors", help="Write benchmark collection")
    
    # Test parameters
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations per test")
    parser.add_argument("--load-duration", type=int, default=120, help="Load test duration in seconds")
    
    # Test selection flags
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--read", action="store_true", help="Run read benchmark only")
    parser.add_argument("--write", action="store_true", help="Run write benchmark only")
    parser.add_argument("--postgres", action="store_true", help="Run PostgreSQL benchmark only")
    parser.add_argument("--postgres-ts", action="store_true", help="Run TimescaleDB benchmark only")
    parser.add_argument("--comparison", action="store_true", help="Run database comparison only")
    parser.add_argument("--load-test", action="store_true", help="Run load test only")
    parser.add_argument("--milvus", action="store_true", help="Run Milvus benchmark only")
    parser.add_argument("--weaviate", action="store_true", help="Run Weaviate benchmark only")
    parser.add_argument("--all-databases", action="store_true", help="Run all database benchmarks (Qdrant, PostgreSQL, TimescaleDB, Milvus, Weaviate)")
    
    # Output options
    parser.add_argument("--output", default="comprehensive_benchmark_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.all or not any([args.read, args.write, args.postgres, args.postgres_ts, args.comparison, args.load_test, args.milvus, args.weaviate, args.all_databases]):
        # If no specific tests are selected, run all
        run_read = run_write = run_postgres = run_postgres_ts = run_comparison = run_load_test = True
        run_milvus = run_weaviate = False
    elif args.all_databases:
        # Run all database benchmarks
        run_read = run_write = run_postgres = run_postgres_ts = run_comparison = run_load_test = True
        run_milvus = run_weaviate = True
    else:
        run_read = args.read
        run_write = args.write
        run_postgres = args.postgres
        run_postgres_ts = args.postgres_ts
        run_comparison = args.comparison
        run_load_test = args.load_test
        run_milvus = args.milvus
        run_weaviate = args.weaviate
    
    benchmark = ComprehensiveBenchmarkSuite(
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
        results = benchmark.run_comprehensive_benchmark(
            read_collection=args.read_collection,
            write_collection=args.write_collection,
            iterations=args.iterations,
            load_duration=args.load_duration,
            run_read=run_read,
            run_write=run_write,
            run_postgres=run_postgres,
            run_postgres_ts=run_postgres_ts,
            run_comparison=run_comparison,
            run_load_test=run_load_test,
            run_milvus=run_milvus,
            run_weaviate=run_weaviate
        )
        benchmark.save_results(results, args.output)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise


if __name__ == "__main__":
    main()

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
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchRequest
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
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


class ComprehensiveBenchmarkSuite:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, 
                 postgres_host="localhost", postgres_port=5432,
                 postgres_user="postgres", postgres_password="postgres",
                 postgres_db="vectordb",
                 milvus_host="localhost", milvus_port=19530,
                 weaviate_host="localhost", weaviate_port=8080,
                 vespa_host="localhost", vespa_port=8081):
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
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.weaviate_host = weaviate_host
        self.weaviate_port = weaviate_port
        self.vespa_host = vespa_host
        self.vespa_port = vespa_port
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
            self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
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
        print("\n2. Batch Vector Search (10 vectors, 10 results each)")
        batch_search_times = []
        batch_iterations = max(1, iterations // 10)
        for _ in tqdm(range(batch_iterations), desc="Batch searches"):
            query_vectors = [self.generate_query_vector() for _ in range(10)]
            start_time = time.time()
            self.qdrant_client.search_batch(
                collection_name=collection_name,
                requests=[SearchRequest(vector=v, limit=10) for v in query_vectors]
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
            self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
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
        print("\n6. Concurrent Searches (100 queries, 10 workers)")
        def single_search():
            query_vector = self.generate_query_vector()
            start_time = time.time()
            self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=10
            )
            return time.time() - start_time
        
        concurrent_times = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(single_search) for _ in range(100)]
            for future in tqdm(as_completed(futures), total=100, desc="Concurrent searches"):
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
        print("\n3. Concurrent Insertions (10 workers, batch size 100)")
        def batch_insert():
            points = [self.generate_test_point(self.next_id + i) for i in range(100)]
            self.next_id += 100
            start_time = time.time()
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            return time.time() - start_time
        
        concurrent_times = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(batch_insert) for _ in range(50)]
            for future in tqdm(as_completed(futures), total=50, desc="Concurrent inserts"):
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
            
            postgres_results['search'] = {
                'mean': statistics.mean(search_times),
                'median': statistics.median(search_times),
                'p95': np.percentile(search_times, 95),
                'p99': np.percentile(search_times, 99),
                'min': min(search_times),
                'max': max(search_times),
                'qps': 1.0 / statistics.mean(search_times)
            }
            
            # Insert performance
            print("\n2. PostgreSQL Insert Performance")
            insert_times = []
            for i in tqdm(range(iterations), desc="PostgreSQL inserts"):
                vector = self.generate_query_vector()
                unique_id = 30000 + i  # Simple unique ID range
                
                start_time = time.time()
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                        VALUES (%s, %s::vector, %s, %s);
                    """, (unique_id, vector, f"Test point {i}", '{"test": true}'))
                    conn.commit()
                insert_times.append(time.time() - start_time)
            
            postgres_results['insert'] = {
                'mean': statistics.mean(insert_times),
                'median': statistics.median(insert_times),
                'p95': np.percentile(insert_times, 95),
                'p99': np.percentile(insert_times, 99),
                'min': min(insert_times),
                'max': max(insert_times),
                'throughput': 1.0 / statistics.mean(insert_times)
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
        
        # Run Qdrant benchmark
        qdrant_read = self.run_read_benchmark(qdrant_collection, iterations)
        qdrant_write = self.run_write_benchmark(qdrant_collection, iterations, cleanup=False)
        
        # Run PostgreSQL benchmark
        postgres_results = self.run_postgres_benchmark(iterations)
        
        if not all([qdrant_read, postgres_results]):
            print("Error: Could not complete database comparison")
            return None
        
        # Calculate performance ratios
        comparison_results = {
            'qdrant_read': qdrant_read,
            'qdrant_write': qdrant_write,
            'postgres': postgres_results,
            'ratios': {}
        }
        
        # Search comparison
        if 'single_search' in qdrant_read and 'search' in postgres_results:
            qdrant_qps = qdrant_read['single_search']['qps']
            postgres_qps = postgres_results['search']['qps']
            if postgres_qps > 0:
                comparison_results['ratios']['search_qps'] = {
                    'qdrant_vs_postgres': qdrant_qps / postgres_qps,
                    'qdrant_qps': qdrant_qps,
                    'postgres_qps': postgres_qps
                }
        
        # Insert comparison
        if 'single_insert' in qdrant_write and 'insert' in postgres_results:
            qdrant_throughput = qdrant_write['single_insert']['throughput']
            postgres_throughput = postgres_results['insert']['throughput']
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
                    self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
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
    
    # ==================== NEW DATABASE BENCHMARKS ====================
    
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
            if not utility.has_collection(collection_name):
                return {"error": f"Collection '{collection_name}' does not exist"}
            
            collection = Collection(collection_name)
            collection.load()
            
            # Read benchmark
            read_times = []
            for _ in tqdm(range(iterations), desc="Milvus reads"):
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
                read_times.append(end_time - start_time)
            
            # Write benchmark
            write_times = []
            for i in tqdm(range(iterations), desc="Milvus writes"):
                vector = self.generate_query_vector()
                unique_id = 50000 + i  # Use different ID range
                
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
                write_times.append(end_time - start_time)
            
            connections.disconnect(connection_alias)
            
            return {
                "read_performance": {
                    "mean": statistics.mean(read_times),
                    "median": statistics.median(read_times),
                    "p95": np.percentile(read_times, 95),
                    "p99": np.percentile(read_times, 99),
                    "qps": 1.0 / statistics.mean(read_times)
                },
                "write_performance": {
                    "mean": statistics.mean(write_times),
                    "median": statistics.median(write_times),
                    "p95": np.percentile(write_times, 95),
                    "p99": np.percentile(write_times, 99),
                    "qps": 1.0 / statistics.mean(write_times)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_weaviate_benchmark(self, class_name: str = "TestVectors", iterations: int = 100):
        """Run Weaviate benchmark (read and write operations)"""
        if not WEAVIATE_AVAILABLE:
            return {"error": "weaviate-client not available. Install with: pip install weaviate-client"}
        
        try:
            # Connect to Weaviate
            client = weaviate.Client(
                url=f"http://{self.weaviate_host}:{self.weaviate_port}",
                additional_headers={"X-OpenAI-Api-Key": "dummy"}
            )
            
            if not client.is_ready():
                return {"error": "Weaviate server not ready"}
            
            if not client.schema.exists(class_name):
                return {"error": f"Class '{class_name}' does not exist"}
            
            # Read benchmark
            read_times = []
            for _ in tqdm(range(iterations), desc="Weaviate reads"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                # Search in Weaviate
                result = client.query.get(class_name, ["text_content", "metadata"]).with_near_vector({
                    "vector": query_vector
                }).with_limit(10).do()
                
                end_time = time.time()
                read_times.append(end_time - start_time)
            
            # Write benchmark
            write_times = []
            for i in tqdm(range(iterations), desc="Weaviate writes"):
                vector = self.generate_query_vector()
                unique_id = f"doc_{60000 + i}"
                
                start_time = time.time()
                
                # Insert into Weaviate
                data_object = {
                    "text_content": f"Test document {unique_id}",
                    "metadata": f'{{"source": "test", "id": "{unique_id}"}}'
                }
                
                client.data_object.create(
                    data_object=data_object,
                    class_name=class_name,
                    vector=vector
                )
                
                end_time = time.time()
                write_times.append(end_time - start_time)
            
            return {
                "read_performance": {
                    "mean": statistics.mean(read_times),
                    "median": statistics.median(read_times),
                    "p95": np.percentile(read_times, 95),
                    "p99": np.percentile(read_times, 99),
                    "qps": 1.0 / statistics.mean(read_times)
                },
                "write_performance": {
                    "mean": statistics.mean(write_times),
                    "median": statistics.median(write_times),
                    "p95": np.percentile(write_times, 95),
                    "p99": np.percentile(write_times, 99),
                    "qps": 1.0 / statistics.mean(write_times)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_vespa_benchmark(self, application_name: str = "test_vectors", document_type: str = "test_vector", iterations: int = 100):
        """Run Vespa benchmark (read and write operations)"""
        try:
            # Test connection
            response = requests.get(f"http://{self.vespa_host}:{self.vespa_port}/ApplicationStatus", timeout=10)
            if response.status_code != 200:
                return {"error": f"Vespa server not ready (status: {response.status_code})"}
            
            # Read benchmark
            read_times = []
            for _ in tqdm(range(iterations), desc="Vespa reads"):
                query_vector = self.generate_query_vector()
                start_time = time.time()
                
                # Search in Vespa
                search_url = f"http://{self.vespa_host}:{self.vespa_port}/search/"
                params = {
                    "yql": f"select * from {document_type} where {document_type} contains 'test'",
                    "hits": 10,
                    "ranking": "cosine_similarity"
                }
                
                response = requests.get(search_url, params=params, timeout=30)
                
                end_time = time.time()
                read_times.append(end_time - start_time)
            
            # Write benchmark
            write_times = []
            for i in tqdm(range(iterations), desc="Vespa writes"):
                vector = self.generate_query_vector()
                unique_id = f"doc_{70000 + i}"
                
                start_time = time.time()
                
                # Insert into Vespa
                document = {
                    "fields": {
                        "id": unique_id,
                        "text_content": f"Test document {unique_id}",
                        "metadata": f'{{"source": "test", "id": "{unique_id}"}}',
                        "vector": {
                            "values": vector
                        }
                    }
                }
                
                url = f"http://{self.vespa_host}:{self.vespa_port}/document/v1/{application_name}/{document_type}/docid/{unique_id}"
                response = requests.put(
                    url,
                    json=document,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                end_time = time.time()
                write_times.append(end_time - start_time)
            
            return {
                "read_performance": {
                    "mean": statistics.mean(read_times),
                    "median": statistics.median(read_times),
                    "p95": np.percentile(read_times, 95),
                    "p99": np.percentile(read_times, 99),
                    "qps": 1.0 / statistics.mean(read_times)
                },
                "write_performance": {
                    "mean": statistics.mean(write_times),
                    "median": statistics.median(write_times),
                    "p95": np.percentile(write_times, 95),
                    "p99": np.percentile(write_times, 99),
                    "qps": 1.0 / statistics.mean(write_times)
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
                                  run_comparison: bool = True,
                                  run_load_test: bool = True,
                                  run_milvus: bool = False,
                                  run_weaviate: bool = False,
                                  run_vespa: bool = False):
        """Run comprehensive benchmark suite based on flags"""
        
        print("Comprehensive Vector Database Benchmark Suite")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Qdrant Host: {self.qdrant_host}:{self.qdrant_port}")
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
                "system_info": system_info
            },
            "read_benchmark": None,
            "write_benchmark": None,
            "postgres_benchmark": None,
            "database_comparison": None,
            "load_test": None,
            "milvus_benchmark": None,
            "weaviate_benchmark": None,
            "vespa_benchmark": None
        }
        
        try:
            # Run read benchmark
            if run_read:
                results["read_benchmark"] = self.run_read_benchmark(read_collection, iterations)
            
            # Run write benchmark
            if run_write:
                # Don't cleanup if using the same collection as read benchmark
                cleanup_write = (write_collection != read_collection)
                results["write_benchmark"] = self.run_write_benchmark(write_collection, iterations, cleanup=cleanup_write)
            
            # Run PostgreSQL benchmark
            if run_postgres:
                results["postgres_benchmark"] = self.run_postgres_benchmark(iterations)
            
            # Run database comparison
            if run_comparison:
                results["database_comparison"] = self.run_database_comparison(read_collection, iterations)
            
            # Run load test
            if run_load_test:
                results["load_test"] = self.run_load_test(read_collection, write_collection, load_duration)
            
            # Run Milvus benchmark
            if run_milvus:
                print("\n" + "="*60)
                print("RUNNING MILVUS BENCHMARK")
                print("="*60)
                results["milvus_benchmark"] = self.run_milvus_benchmark(read_collection, iterations)
            
            # Run Weaviate benchmark
            if run_weaviate:
                print("\n" + "="*60)
                print("RUNNING WEAVIATE BENCHMARK")
                print("="*60)
                results["weaviate_benchmark"] = self.run_weaviate_benchmark("TestVectors", iterations)
            
            # Run Vespa benchmark
            if run_vespa:
                print("\n" + "="*60)
                print("RUNNING VESPA BENCHMARK")
                print("="*60)
                results["vespa_benchmark"] = self.run_vespa_benchmark("test_vectors", "test_vector", iterations)
            
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
                print(f"  Read: {milvus_results['read_performance']['mean']:.4f}s mean, {milvus_results['read_performance']['qps']:.2f} QPS")
                print(f"  Write: {milvus_results['write_performance']['mean']:.4f}s mean, {milvus_results['write_performance']['qps']:.2f} QPS")
        
        if results["weaviate_benchmark"]:
            print("\nWEAVIATE PERFORMANCE:")
            weaviate_results = results["weaviate_benchmark"]
            if "error" in weaviate_results:
                print(f"  Error: {weaviate_results['error']}")
            else:
                print(f"  Read: {weaviate_results['read_performance']['mean']:.4f}s mean, {weaviate_results['read_performance']['qps']:.2f} QPS")
                print(f"  Write: {weaviate_results['write_performance']['mean']:.4f}s mean, {weaviate_results['write_performance']['qps']:.2f} QPS")
        
        if results["vespa_benchmark"]:
            print("\nVESPA PERFORMANCE:")
            vespa_results = results["vespa_benchmark"]
            if "error" in vespa_results:
                print(f"  Error: {vespa_results['error']}")
            else:
                print(f"  Read: {vespa_results['read_performance']['mean']:.4f}s mean, {vespa_results['read_performance']['qps']:.2f} QPS")
                print(f"  Write: {vespa_results['write_performance']['mean']:.4f}s mean, {vespa_results['write_performance']['qps']:.2f} QPS")
        
        # Add comprehensive performance comparison summary
        self.print_performance_comparison_summary(results)
    
    def print_performance_comparison_summary(self, results):
        """Print comprehensive performance comparison between Qdrant and PostgreSQL"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        
        # Extract performance data
        qdrant_read = results.get("read_benchmark", {})
        qdrant_write = results.get("write_benchmark", {})
        postgres = results.get("postgres_benchmark", {})
        
        if not all([qdrant_read, postgres]):
            print("Insufficient data for performance comparison")
            return
        
        print("\n🔍 SEARCH PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        # Single search comparison
        if "single_search" in qdrant_read and "search" in postgres:
            qdrant_qps = qdrant_read["single_search"]["qps"]
            postgres_qps = postgres["search"]["qps"]
            ratio = qdrant_qps / postgres_qps
            winner = "Qdrant" if ratio > 1 else "PostgreSQL"
            print(f"Single Search:")
            print(f"  Qdrant:     {qdrant_qps:.1f} QPS ({qdrant_read['single_search']['mean']:.4f}s)")
            print(f"  PostgreSQL: {postgres_qps:.1f} QPS ({postgres['search']['mean']:.4f}s)")
            print(f"  🏆 {winner} is {max(ratio, 1/ratio):.1f}x faster")
        
        # Batch search comparison (if available)
        if "batch_search" in qdrant_read:
            qdrant_batch_qps = qdrant_read["batch_search"]["qps"]
            print(f"\nBatch Search (10 vectors):")
            print(f"  Qdrant:     {qdrant_batch_qps:.1f} QPS ({qdrant_read['batch_search']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested in batch mode)")
        
        # Filtered search comparison
        if "filtered_search" in qdrant_read:
            qdrant_filtered_qps = qdrant_read["filtered_search"]["qps"]
            print(f"\nFiltered Search:")
            print(f"  Qdrant:     {qdrant_filtered_qps:.1f} QPS ({qdrant_read['filtered_search']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested with filters)")
        
        # ID retrieval comparison
        if "retrieve_by_id" in qdrant_read:
            qdrant_id_qps = qdrant_read["retrieve_by_id"]["qps"]
            print(f"\nID Retrieval:")
            print(f"  Qdrant:     {qdrant_id_qps:.1f} QPS ({qdrant_read['retrieve_by_id']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested)")
        
        print("\n✏️  WRITE PERFORMANCE COMPARISON:")
        print("-" * 50)
        
        # Single insert comparison
        if "single_insert" in qdrant_write and "insert" in postgres:
            qdrant_insert_ops = qdrant_write["single_insert"]["throughput"]
            postgres_insert_ops = postgres["insert"]["throughput"]
            ratio = qdrant_insert_ops / postgres_insert_ops
            winner = "Qdrant" if ratio > 1 else "PostgreSQL"
            print(f"Single Insert:")
            print(f"  Qdrant:     {qdrant_insert_ops:.1f} ops/sec ({qdrant_write['single_insert']['mean']:.4f}s)")
            print(f"  PostgreSQL: {postgres_insert_ops:.1f} ops/sec ({postgres['insert']['mean']:.4f}s)")
            print(f"  🏆 {winner} is {max(ratio, 1/ratio):.1f}x faster")
        
        # Batch insert performance (Qdrant only)
        if "batch_insert_10" in qdrant_write:
            qdrant_batch_10 = qdrant_write["batch_insert_10"]["throughput"]
            print(f"\nBatch Insert (10 points):")
            print(f"  Qdrant:     {qdrant_batch_10:.1f} points/sec ({qdrant_write['batch_insert_10']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested in batch mode)")
        
        if "batch_insert_100" in qdrant_write:
            qdrant_batch_100 = qdrant_write["batch_insert_100"]["throughput"]
            print(f"\nBatch Insert (100 points):")
            print(f"  Qdrant:     {qdrant_batch_100:.1f} points/sec ({qdrant_write['batch_insert_100']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested in batch mode)")
        
        if "batch_insert_1000" in qdrant_write:
            qdrant_batch_1000 = qdrant_write["batch_insert_1000"]["throughput"]
            print(f"\nBatch Insert (1000 points):")
            print(f"  Qdrant:     {qdrant_batch_1000:.1f} points/sec ({qdrant_write['batch_insert_1000']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested in batch mode)")
        
        # Update and delete operations (Qdrant only)
        if "update" in qdrant_write:
            qdrant_update_ops = qdrant_write["update"]["throughput"]
            print(f"\nUpdate Operations:")
            print(f"  Qdrant:     {qdrant_update_ops:.1f} ops/sec ({qdrant_write['update']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested)")
        
        if "delete" in qdrant_write:
            qdrant_delete_ops = qdrant_write["delete"]["throughput"]
            print(f"\nDelete Operations:")
            print(f"  Qdrant:     {qdrant_delete_ops:.1f} ops/sec ({qdrant_write['delete']['mean']:.4f}s)")
            print(f"  PostgreSQL: N/A (not tested)")
        
        print("\n📊 OVERALL PERFORMANCE INSIGHTS:")
        print("-" * 50)
        
        # Calculate overall performance ratios
        if "single_search" in qdrant_read and "search" in postgres and "single_insert" in qdrant_write and "insert" in postgres:
            search_ratio = qdrant_read["single_search"]["qps"] / postgres["search"]["qps"]
            insert_ratio = qdrant_write["single_insert"]["throughput"] / postgres["insert"]["throughput"]
            
            print(f"• Search Performance: Qdrant is {search_ratio:.1f}x {'faster' if search_ratio > 1 else 'slower'} than PostgreSQL")
            print(f"• Insert Performance: Qdrant is {insert_ratio:.1f}x {'faster' if insert_ratio > 1 else 'slower'} than PostgreSQL")
            
            # Overall winner
            if search_ratio > 1 and insert_ratio > 1:
                print(f"• 🏆 Overall Winner: Qdrant excels in both search and insert operations")
            elif search_ratio < 1 and insert_ratio < 1:
                print(f"• 🏆 Overall Winner: PostgreSQL excels in both search and insert operations")
            else:
                print(f"• 🤝 Mixed Results: Each database excels in different areas")
        
        # Additional insights
        if "concurrent_search" in qdrant_read:
            concurrent_qps = qdrant_read["concurrent_search"]["qps"]
            print(f"• Concurrent Search: Qdrant handles {concurrent_qps:.1f} QPS under load")
        
        if "scroll" in qdrant_read:
            scroll_qps = qdrant_read["scroll"]["qps"]
            print(f"• Large Dataset Scrolling: Qdrant processes {scroll_qps:.1f} QPS for bulk operations")
        
        # Memory and CPU insights
        if results.get("load_test"):
            load_stats = results["load_test"]
            print(f"• System Load: {load_stats['cpu_usage']['mean']:.1f}% CPU, {load_stats['memory_usage']['mean']:.1f}% Memory during sustained load")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        
        if "single_search" in qdrant_read and "search" in postgres:
            search_ratio = qdrant_read["single_search"]["qps"] / postgres["search"]["qps"]
            if search_ratio > 1.5:
                print("• Use Qdrant for high-performance search applications")
            elif search_ratio < 0.7:
                print("• Consider PostgreSQL for search if you need SQL integration")
            else:
                print("• Both databases perform similarly for search - choose based on other requirements")
        
        if "single_insert" in qdrant_write and "insert" in postgres:
            insert_ratio = qdrant_write["single_insert"]["throughput"] / postgres["insert"]["throughput"]
            if insert_ratio > 2:
                print("• Use Qdrant for high-throughput vector insertions")
            elif insert_ratio < 0.5:
                print("• Consider PostgreSQL for insert-heavy workloads if you need SQL features")
            else:
                print("• Both databases handle inserts well - consider your data management needs")
        
        print("• Use Qdrant for:")
        print("  - Real-time vector search applications")
        print("  - High-throughput batch operations")
        print("  - Applications requiring advanced vector operations (filtering, scrolling)")
        print("  - Microservices architectures")
        
        print("• Use PostgreSQL with pgvector for:")
        print("  - Applications requiring SQL integration")
        print("  - Complex relational queries with vector search")
        print("  - Existing PostgreSQL ecosystems")
        print("  - ACID compliance requirements")
    
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
    parser.add_argument("--milvus-host", default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--weaviate-host", default="localhost", help="Weaviate host")
    parser.add_argument("--weaviate-port", type=int, default=8080, help="Weaviate port")
    parser.add_argument("--vespa-host", default="localhost", help="Vespa host")
    parser.add_argument("--vespa-port", type=int, default=8081, help="Vespa port")
    
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
    parser.add_argument("--comparison", action="store_true", help="Run database comparison only")
    parser.add_argument("--load-test", action="store_true", help="Run load test only")
    parser.add_argument("--milvus", action="store_true", help="Run Milvus benchmark only")
    parser.add_argument("--weaviate", action="store_true", help="Run Weaviate benchmark only")
    parser.add_argument("--vespa", action="store_true", help="Run Vespa benchmark only")
    parser.add_argument("--all-databases", action="store_true", help="Run all database benchmarks (Qdrant, PostgreSQL, Milvus, Weaviate, Vespa)")
    
    # Output options
    parser.add_argument("--output", default="comprehensive_benchmark_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Determine which tests to run
    if args.all or not any([args.read, args.write, args.postgres, args.comparison, args.load_test, args.milvus, args.weaviate, args.vespa, args.all_databases]):
        # If no specific tests are selected, run all
        run_read = run_write = run_postgres = run_comparison = run_load_test = True
        run_milvus = run_weaviate = run_vespa = False
    elif args.all_databases:
        # Run all database benchmarks
        run_read = run_write = run_postgres = run_comparison = run_load_test = True
        run_milvus = run_weaviate = run_vespa = True
    else:
        run_read = args.read
        run_write = args.write
        run_postgres = args.postgres
        run_comparison = args.comparison
        run_load_test = args.load_test
        run_milvus = args.milvus
        run_weaviate = args.weaviate
        run_vespa = args.vespa
    
    benchmark = ComprehensiveBenchmarkSuite(
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
        results = benchmark.run_comprehensive_benchmark(
            read_collection=args.read_collection,
            write_collection=args.write_collection,
            iterations=args.iterations,
            load_duration=args.load_duration,
            run_read=run_read,
            run_write=run_write,
            run_postgres=run_postgres,
            run_comparison=run_comparison,
            run_load_test=run_load_test,
            run_milvus=run_milvus,
            run_weaviate=run_weaviate,
            run_vespa=run_vespa
        )
        benchmark.save_results(results, args.output)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise


if __name__ == "__main__":
    main()

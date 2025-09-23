#!/usr/bin/env python3
"""
Vector Database Read Performance Benchmark
Tests various read operations on the populated vector database.
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from tqdm import tqdm
import psutil
import os
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class ReadBenchmark:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "test_vectors"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_dim = self._get_collection_vector_dim()
        self.results = {}
        
    def _get_collection_vector_dim(self) -> int:
        """Get the vector dimension from the collection configuration"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.config.params.vectors.size
        except Exception as e:
            print(f"Warning: Could not get vector dimension from collection, using default 1024: {e}")
            return 1024
        
    def generate_query_vector(self) -> List[float]:
        """Generate a random query vector"""
        vector = np.random.normal(0, 1, self.vector_dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def benchmark_single_search(self, limit: int = 10) -> float:
        """Benchmark single vector search"""
        query_vector = self.generate_query_vector()
        
        start_time = time.time()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        end_time = time.time()
        
        return end_time - start_time

    def benchmark_batch_search(self, batch_size: int = 10, limit: int = 10) -> float:
        """Benchmark batch vector search"""
        query_vectors = [self.generate_query_vector() for _ in range(batch_size)]
        
        start_time = time.time()
        results = self.client.search_batch(
            collection_name=self.collection_name,
            requests=[
                SearchRequest(vector=vector, limit=limit)
                for vector in query_vectors
            ]
        )
        end_time = time.time()
        
        return end_time - start_time

    def benchmark_filtered_search(self, limit: int = 10) -> float:
        """Benchmark filtered vector search"""
        query_vector = self.generate_query_vector()
        
        # Create a random filter
        categories = ["technical", "business", "legal", "scientific"]
        random_category = random.choice(categories)
        
        start_time = time.time()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.category",
                        match=MatchValue(value=random_category)
                    )
                ]
            ),
            limit=limit
        )
        end_time = time.time()
        
        return end_time - start_time

    def benchmark_retrieve_by_id(self, point_ids: List[int]) -> float:
        """Benchmark retrieving points by ID"""
        start_time = time.time()
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids
        )
        end_time = time.time()
        
        return end_time - start_time

    def benchmark_scroll(self, limit: int = 1000) -> float:
        """Benchmark scrolling through collection"""
        start_time = time.time()
        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        end_time = time.time()
        
        return end_time - start_time

    def run_concurrent_searches(self, num_queries: int = 100, max_workers: int = 10) -> List[float]:
        """Run concurrent search operations"""
        def single_search():
            return self.benchmark_single_search()
        
        times = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_search) for _ in range(num_queries)]
            
            for future in tqdm(as_completed(futures), total=num_queries, desc="Concurrent searches"):
                times.append(future.result())
        
        return times

    def get_random_point_ids(self, count: int) -> List[int]:
        """Get random point IDs from the collection"""
        try:
            # Get collection info to know the range of IDs
            collection_info = self.client.get_collection(self.collection_name)
            max_id = collection_info.points_count
            
            # Generate random IDs (assuming IDs start from 0)
            return [random.randint(0, max_id - 1) for _ in range(count)]
        except Exception as e:
            print(f"Error getting random point IDs: {e}")
            return list(range(count))

    def run_benchmark_suite(self, iterations: int = 100):
        """Run comprehensive benchmark suite"""
        print("Starting Read Performance Benchmark Suite")
        print("="*50)
        
        # Get collection stats
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"Collection: {self.collection_name}")
            print(f"Total points: {collection_info.points_count:,}")
            print(f"Vector dimension: {self.vector_dim}")
            print()
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return

        # 1. Single Vector Search
        print("1. Single Vector Search (10 results)")
        single_search_times = []
        for _ in tqdm(range(iterations), desc="Single searches"):
            single_search_times.append(self.benchmark_single_search(limit=10))
        
        self.results['single_search'] = {
            'times': single_search_times,
            'mean': statistics.mean(single_search_times),
            'median': statistics.median(single_search_times),
            'p95': np.percentile(single_search_times, 95),
            'p99': np.percentile(single_search_times, 99),
            'min': min(single_search_times),
            'max': max(single_search_times)
        }

        # 2. Batch Search
        print("\n2. Batch Vector Search (10 vectors, 10 results each)")
        batch_search_times = []
        batch_iterations = max(1, iterations // 10)  # Ensure at least 1 iteration
        for _ in tqdm(range(batch_iterations), desc="Batch searches"):  # Fewer iterations for batch
            batch_search_times.append(self.benchmark_batch_search(batch_size=10, limit=10))
        
        self.results['batch_search'] = {
            'times': batch_search_times,
            'mean': statistics.mean(batch_search_times),
            'median': statistics.median(batch_search_times),
            'p95': np.percentile(batch_search_times, 95),
            'p99': np.percentile(batch_search_times, 99),
            'min': min(batch_search_times),
            'max': max(batch_search_times)
        }

        # 3. Filtered Search
        print("\n3. Filtered Vector Search")
        filtered_search_times = []
        for _ in tqdm(range(iterations), desc="Filtered searches"):
            filtered_search_times.append(self.benchmark_filtered_search(limit=10))
        
        self.results['filtered_search'] = {
            'times': filtered_search_times,
            'mean': statistics.mean(filtered_search_times),
            'median': statistics.median(filtered_search_times),
            'p95': np.percentile(filtered_search_times, 95),
            'p99': np.percentile(filtered_search_times, 99),
            'min': min(filtered_search_times),
            'max': max(filtered_search_times)
        }

        # 4. Retrieve by ID
        print("\n4. Retrieve by ID (10 points)")
        retrieve_times = []
        point_ids = self.get_random_point_ids(10)
        for _ in tqdm(range(iterations), desc="ID retrievals"):
            retrieve_times.append(self.benchmark_retrieve_by_id(point_ids))
        
        self.results['retrieve_by_id'] = {
            'times': retrieve_times,
            'mean': statistics.mean(retrieve_times),
            'median': statistics.median(retrieve_times),
            'p95': np.percentile(retrieve_times, 95),
            'p99': np.percentile(retrieve_times, 99),
            'min': min(retrieve_times),
            'max': max(retrieve_times)
        }

        # 5. Scroll
        print("\n5. Scroll Collection (1000 points)")
        scroll_times = []
        scroll_iterations = max(1, iterations // 10)  # Ensure at least 1 iteration
        for _ in tqdm(range(scroll_iterations), desc="Scroll operations"):  # Fewer iterations for scroll
            scroll_times.append(self.benchmark_scroll(limit=1000))
        
        self.results['scroll'] = {
            'times': scroll_times,
            'mean': statistics.mean(scroll_times),
            'median': statistics.median(scroll_times),
            'p95': np.percentile(scroll_times, 95),
            'p99': np.percentile(scroll_times, 99),
            'min': min(scroll_times),
            'max': max(scroll_times)
        }

        # 6. Concurrent Searches
        print("\n6. Concurrent Searches (100 queries, 10 workers)")
        concurrent_times = self.run_concurrent_searches(num_queries=100, max_workers=10)
        
        self.results['concurrent_search'] = {
            'times': concurrent_times,
            'mean': statistics.mean(concurrent_times),
            'median': statistics.median(concurrent_times),
            'p95': np.percentile(concurrent_times, 95),
            'p99': np.percentile(concurrent_times, 99),
            'min': min(concurrent_times),
            'max': max(concurrent_times)
        }

        # Print results
        self.print_results()

    def print_results(self):
        """Print benchmark results in a formatted way"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        for operation, stats in self.results.items():
            print(f"\n{operation.upper().replace('_', ' ')}")
            print("-" * 40)
            print(f"Mean:     {stats['mean']:.4f}s")
            print(f"Median:   {stats['median']:.4f}s")
            print(f"P95:      {stats['p95']:.4f}s")
            print(f"P99:      {stats['p99']:.4f}s")
            print(f"Min:      {stats['min']:.4f}s")
            print(f"Max:      {stats['max']:.4f}s")
            
            # Calculate QPS (Queries Per Second)
            if stats['mean'] > 0:
                qps = 1.0 / stats['mean']
                print(f"QPS:      {qps:.2f}")

    def save_results(self, filename: str = "read_benchmark_results.json"):
        """Save results to JSON file"""
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for operation, stats in self.results.items():
            serializable_results[operation] = {
                'mean': float(stats['mean']),
                'median': float(stats['median']),
                'p95': float(stats['p95']),
                'p99': float(stats['p99']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'qps': float(1.0 / stats['mean']) if stats['mean'] > 0 else 0
            }
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        
        # Ensure filename is in results directory
        if not filename.startswith("results/"):
            filename = f"results/{filename}"
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vector database read performance")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="test_vectors", help="Collection name")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations per test")
    parser.add_argument("--output", default="read_benchmark_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    benchmark = ReadBenchmark(
        host=args.host,
        port=args.port,
        collection_name=args.collection
    )
    
    try:
        benchmark.run_benchmark_suite(iterations=args.iterations)
        benchmark.save_results(args.output)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise


if __name__ == "__main__":
    main()

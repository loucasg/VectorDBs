#!/usr/bin/env python3
"""
Vector Database Write Performance Benchmark
Tests various write operations on the vector database.
"""

import asyncio
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, UpdateStatus
from tqdm import tqdm
import psutil
import os
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class WriteBenchmark:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "write_test_vectors"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_dim = self._get_collection_vector_dim()
        self.results = {}
        self.next_id = 0
        
    def _get_collection_vector_dim(self) -> int:
        """Get the vector dimension from the collection configuration"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.config.params.vectors.size
        except Exception as e:
            # If collection doesn't exist, try to get dimension from another collection
            try:
                collections = self.client.get_collections()
                if collections.collections:
                    # Use the first available collection to get vector dimension
                    first_collection = collections.collections[0].name
                    collection_info = self.client.get_collection(first_collection)
                    return collection_info.config.params.vectors.size
            except Exception:
                pass
            
            print(f"Warning: Could not get vector dimension from collection, using default 768: {e}")
            return 768
        
    def create_test_collection(self):
        """Create a test collection for write benchmarks"""
        try:
            # Check if collection exists and delete it
            collections = self.client.get_collections()
            if any(col.name == self.collection_name for col in collections.collections):
                print(f"Deleting existing collection '{self.collection_name}'...")
                self.client.delete_collection(self.collection_name)
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created test collection '{self.collection_name}'")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def generate_random_vector(self) -> List[float]:
        """Generate a random normalized vector"""
        vector = np.random.normal(0, 1, self.vector_dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()

    def generate_test_point(self, point_id: int) -> PointStruct:
        """Generate a test point"""
        vector = self.generate_random_vector()
        payload = {
            "id": point_id,
            "text": f"Test point {point_id}",
            "metadata": {
                "category": random.choice(["A", "B", "C", "D"]),
                "value": random.uniform(0, 100),
                "timestamp": int(time.time())
            }
        }
        
        return PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )

    def benchmark_single_insert(self) -> float:
        """Benchmark single point insertion"""
        point = self.generate_test_point(self.next_id)
        self.next_id += 1
        
        start_time = time.time()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        end_time = time.time()
        
        return end_time - start_time

    def benchmark_batch_insert(self, batch_size: int) -> float:
        """Benchmark batch point insertion"""
        points = [self.generate_test_point(self.next_id + i) for i in range(batch_size)]
        self.next_id += batch_size
        
        start_time = time.time()
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        end_time = time.time()
        
        return end_time - start_time

    def benchmark_concurrent_inserts(self, num_operations: int, batch_size: int, max_workers: int) -> List[float]:
        """Benchmark concurrent insertions"""
        def single_batch_insert():
            return self.benchmark_batch_insert(batch_size)
        
        times = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(single_batch_insert) for _ in range(num_operations)]
            
            for future in tqdm(as_completed(futures), total=num_operations, desc="Concurrent inserts"):
                times.append(future.result())
        
        return times

    def benchmark_update_operations(self, num_updates: int) -> List[float]:
        """Benchmark point update operations"""
        times = []
        
        for _ in tqdm(range(num_updates), desc="Update operations"):
            # Generate a random point ID to update
            point_id = random.randint(0, self.next_id - 1)
            
            # Generate new vector and payload
            new_vector = self.generate_random_vector()
            new_payload = {
                "id": point_id,
                "text": f"Updated point {point_id}",
                "metadata": {
                    "category": random.choice(["A", "B", "C", "D"]),
                    "value": random.uniform(0, 100),
                    "timestamp": int(time.time()),
                    "updated": True
                }
            }
            
            point = PointStruct(
                id=point_id,
                vector=new_vector,
                payload=new_payload
            )
            
            start_time = time.time()
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return times

    def benchmark_delete_operations(self, num_deletes: int) -> List[float]:
        """Benchmark point deletion operations"""
        times = []
        
        # Get some random point IDs to delete
        try:
            collection_info = self.client.get_collection(self.collection_name)
            max_id = collection_info.points_count
            point_ids_to_delete = [random.randint(0, max_id - 1) for _ in range(num_deletes)]
        except:
            point_ids_to_delete = list(range(num_deletes))
        
        for point_id in tqdm(point_ids_to_delete, desc="Delete operations"):
            start_time = time.time()
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return times

    def benchmark_bulk_operations(self, num_batches: int, batch_size: int) -> List[float]:
        """Benchmark bulk operations with different batch sizes"""
        times = []
        
        for _ in tqdm(range(num_batches), desc="Bulk operations"):
            points = [self.generate_test_point(self.next_id + i) for i in range(batch_size)]
            self.next_id += batch_size
            
            start_time = time.time()
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return times

    def run_benchmark_suite(self, iterations: int = 100):
        """Run comprehensive write benchmark suite"""
        print("Starting Write Performance Benchmark Suite")
        print("="*50)
        
        # Create test collection
        self.create_test_collection()
        
        # 1. Single Point Insertions
        print("1. Single Point Insertions")
        single_insert_times = []
        for _ in tqdm(range(iterations), desc="Single inserts"):
            single_insert_times.append(self.benchmark_single_insert())
        
        self.results['single_insert'] = {
            'times': single_insert_times,
            'mean': statistics.mean(single_insert_times),
            'median': statistics.median(single_insert_times),
            'p95': np.percentile(single_insert_times, 95),
            'p99': np.percentile(single_insert_times, 99),
            'min': min(single_insert_times),
            'max': max(single_insert_times)
        }

        # 2. Batch Insertions (different batch sizes)
        batch_sizes = [10, 100, 1000]
        for batch_size in batch_sizes:
            print(f"\n2. Batch Insertions (batch size: {batch_size})")
            batch_insert_times = []
            num_batches = max(1, iterations // batch_size)
            
            for _ in tqdm(range(num_batches), desc=f"Batch {batch_size}"):
                batch_insert_times.append(self.benchmark_batch_insert(batch_size))
            
            self.results[f'batch_insert_{batch_size}'] = {
                'times': batch_insert_times,
                'mean': statistics.mean(batch_insert_times),
                'median': statistics.median(batch_insert_times),
                'p95': np.percentile(batch_insert_times, 95),
                'p99': np.percentile(batch_insert_times, 99),
                'min': min(batch_insert_times),
                'max': max(batch_insert_times),
                'batch_size': batch_size
            }

        # 3. Concurrent Insertions
        print("\n3. Concurrent Insertions (10 workers, batch size 100)")
        concurrent_times = self.benchmark_concurrent_inserts(
            num_operations=50,
            batch_size=100,
            max_workers=10
        )
        
        self.results['concurrent_insert'] = {
            'times': concurrent_times,
            'mean': statistics.mean(concurrent_times),
            'median': statistics.median(concurrent_times),
            'p95': np.percentile(concurrent_times, 95),
            'p99': np.percentile(concurrent_times, 99),
            'min': min(concurrent_times),
            'max': max(concurrent_times)
        }

        # 4. Update Operations
        print("\n4. Update Operations")
        update_times = self.benchmark_update_operations(iterations)
        
        self.results['update'] = {
            'times': update_times,
            'mean': statistics.mean(update_times),
            'median': statistics.median(update_times),
            'p95': np.percentile(update_times, 95),
            'p99': np.percentile(update_times, 99),
            'min': min(update_times),
            'max': max(update_times)
        }

        # 5. Delete Operations
        print("\n5. Delete Operations")
        delete_times = self.benchmark_delete_operations(min(iterations, 100))  # Limit deletes
        
        self.results['delete'] = {
            'times': delete_times,
            'mean': statistics.mean(delete_times),
            'median': statistics.median(delete_times),
            'p95': np.percentile(delete_times, 95),
            'p99': np.percentile(delete_times, 99),
            'min': min(delete_times),
            'max': max(delete_times)
        }

        # Print results
        self.print_results()

    def print_results(self):
        """Print benchmark results in a formatted way"""
        print("\n" + "="*80)
        print("WRITE BENCHMARK RESULTS")
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
            
            # Calculate throughput
            if stats['mean'] > 0:
                if 'batch_size' in stats:
                    throughput = stats['batch_size'] / stats['mean']
                    print(f"Throughput: {throughput:.2f} points/sec")
                else:
                    throughput = 1.0 / stats['mean']
                    print(f"Throughput: {throughput:.2f} ops/sec")

    def save_results(self, filename: str = "write_benchmark_results.json"):
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
                'throughput': float(1.0 / stats['mean']) if stats['mean'] > 0 else 0
            }
            
            if 'batch_size' in stats:
                serializable_results[operation]['batch_size'] = stats['batch_size']
                serializable_results[operation]['throughput'] = float(stats['batch_size'] / stats['mean'])
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        
        # Ensure filename is in results directory
        if not filename.startswith("results/"):
            filename = f"results/{filename}"
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")

    def cleanup(self):
        """Clean up test collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Cleaned up test collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error cleaning up collection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vector database write performance")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="write_test_vectors", help="Collection name")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations per test")
    parser.add_argument("--output", default="write_benchmark_results.json", help="Output file for results")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test collection after benchmark")
    
    args = parser.parse_args()
    
    benchmark = WriteBenchmark(
        host=args.host,
        port=args.port,
        collection_name=args.collection
    )
    
    try:
        benchmark.run_benchmark_suite(iterations=args.iterations)
        benchmark.save_results(args.output)
        
        if args.cleanup:
            benchmark.cleanup()
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if args.cleanup:
            benchmark.cleanup()
    except Exception as e:
        print(f"Error during benchmark: {e}")
        if args.cleanup:
            benchmark.cleanup()
        raise


if __name__ == "__main__":
    main()

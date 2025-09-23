#!/usr/bin/env python3
"""
Vector Database Comparison Script
Compares performance between Qdrant and PostgreSQL with pgvector.
"""

import time
import random
import numpy as np
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import RealDictCursor
import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed


class DatabaseComparison:
    def __init__(self):
        self.vector_dim = 768
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.postgres_conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="vectordb",
            user="postgres",
            password="postgres"
        )
        
    def generate_random_vector(self) -> List[float]:
        """Generate a random normalized vector"""
        vector = np.random.normal(0, 1, self.vector_dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    def test_qdrant_search(self, num_queries: int = 100) -> List[float]:
        """Test Qdrant search performance"""
        times = []
        
        for _ in range(num_queries):
            query_vector = self.generate_random_vector()
            
            start_time = time.time()
            results = self.qdrant_client.search(
                collection_name="test_vectors",
                query_vector=query_vector,
                limit=10
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return times
    
    def test_postgres_search(self, num_queries: int = 100) -> List[float]:
        """Test PostgreSQL search performance"""
        times = []
        
        with self.postgres_conn.cursor() as cur:
            for _ in range(num_queries):
                query_vector = self.generate_random_vector()
                
                start_time = time.time()
                cur.execute("""
                    SELECT vector_id, text_content, metadata, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM vector_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT 10;
                """, (query_vector, query_vector))
                
                results = cur.fetchall()
                end_time = time.time()
                
                times.append(end_time - start_time)
        
        return times
    
    def test_qdrant_batch_search(self, num_batches: int = 10, batch_size: int = 10) -> List[float]:
        """Test Qdrant batch search performance"""
        times = []
        
        for _ in range(num_batches):
            query_vectors = [self.generate_random_vector() for _ in range(batch_size)]
            
            start_time = time.time()
            results = self.qdrant_client.search_batch(
                collection_name="test_vectors",
                requests=[{"vector": v, "limit": 10} for v in query_vectors]
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return times
    
    def test_postgres_batch_search(self, num_batches: int = 10, batch_size: int = 10) -> List[float]:
        """Test PostgreSQL batch search performance"""
        times = []
        
        with self.postgres_conn.cursor() as cur:
            for _ in range(num_batches):
                query_vectors = [self.generate_random_vector() for _ in range(batch_size)]
                
                start_time = time.time()
                for query_vector in query_vectors:
                    cur.execute("""
                        SELECT vector_id, text_content, metadata, 
                               1 - (embedding <=> %s::vector) as similarity
                        FROM vector_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, query_vector))
                    cur.fetchall()
                
                end_time = time.time()
                times.append(end_time - start_time)
        
        return times
    
    def test_concurrent_searches(self, num_queries: int = 100, max_workers: int = 10):
        """Test concurrent search performance"""
        def qdrant_search():
            query_vector = self.generate_random_vector()
            start_time = time.time()
            self.qdrant_client.search(
                collection_name="test_vectors",
                query_vector=query_vector,
                limit=10
            )
            return time.time() - start_time
        
        def postgres_search():
            query_vector = self.generate_random_vector()
            start_time = time.time()
            with self.postgres_conn.cursor() as cur:
                cur.execute("""
                    SELECT vector_id, text_content, metadata, 
                           1 - (embedding <=> %s::vector) as similarity
                    FROM vector_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT 10;
                """, (query_vector, query_vector))
                cur.fetchall()
            return time.time() - start_time
        
        # Test Qdrant concurrent searches
        qdrant_times = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(qdrant_search) for _ in range(num_queries)]
            for future in as_completed(futures):
                qdrant_times.append(future.result())
        
        # Test PostgreSQL concurrent searches
        postgres_times = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(postgres_search) for _ in range(num_queries)]
            for future in as_completed(futures):
                postgres_times.append(future.result())
        
        return qdrant_times, postgres_times
    
    def get_database_stats(self):
        """Get statistics from both databases"""
        stats = {}
        
        # Qdrant stats
        try:
            collection_info = self.qdrant_client.get_collection("test_vectors")
            stats['qdrant'] = {
                'points_count': collection_info.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance,
                'indexed_vectors': collection_info.indexed_vectors_count
            }
        except Exception as e:
            stats['qdrant'] = {'error': str(e)}
        
        # PostgreSQL stats
        try:
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT COUNT(*) as total_points FROM vector_embeddings;")
                total_points = cur.fetchone()['total_points']
                
                cur.execute("""
                    SELECT 
                        AVG(LENGTH(metadata::TEXT)) as avg_metadata_size,
                        MIN(created_at) as earliest_record,
                        MAX(created_at) as latest_record
                    FROM vector_embeddings;
                """)
                metadata_stats = cur.fetchone()
                
                stats['postgres'] = {
                    'points_count': total_points,
                    'vector_size': self.vector_dim,
                    'avg_metadata_size': metadata_stats['avg_metadata_size'],
                    'earliest_record': metadata_stats['earliest_record'],
                    'latest_record': metadata_stats['latest_record']
                }
        except Exception as e:
            stats['postgres'] = {'error': str(e)}
        
        return stats
    
    def run_comparison(self, num_queries: int = 100, num_batches: int = 10, batch_size: int = 10):
        """Run comprehensive comparison between databases"""
        print("Vector Database Performance Comparison")
        print("=" * 50)
        print(f"Test parameters:")
        print(f"  Single queries: {num_queries}")
        print(f"  Batch queries: {num_batches} batches of {batch_size}")
        print(f"  Vector dimension: {self.vector_dim}")
        print()
        
        # Get database stats
        print("Database Statistics:")
        stats = self.get_database_stats()
        
        if 'error' not in stats['qdrant']:
            print(f"Qdrant: {stats['qdrant']['points_count']:,} points, {stats['qdrant']['vector_size']}D vectors")
        else:
            print(f"Qdrant: Error - {stats['qdrant']['error']}")
        
        if 'error' not in stats['postgres']:
            print(f"PostgreSQL: {stats['postgres']['points_count']:,} points, {stats['postgres']['vector_size']}D vectors")
        else:
            print(f"PostgreSQL: Error - {stats['postgres']['error']}")
        
        print()
        
        # Single search performance
        print("1. Single Search Performance")
        print("-" * 30)
        
        if 'error' not in stats['qdrant']:
            print("Testing Qdrant single searches...")
            qdrant_single_times = self.test_qdrant_search(num_queries)
            qdrant_single_stats = {
                'mean': statistics.mean(qdrant_single_times),
                'median': statistics.median(qdrant_single_times),
                'p95': np.percentile(qdrant_single_times, 95),
                'p99': np.percentile(qdrant_single_times, 99),
                'qps': 1.0 / statistics.mean(qdrant_single_times)
            }
            print(f"  Mean: {qdrant_single_stats['mean']:.4f}s")
            print(f"  QPS: {qdrant_single_stats['qps']:.2f}")
        else:
            qdrant_single_stats = None
            print("  Qdrant: Not available")
        
        if 'error' not in stats['postgres']:
            print("Testing PostgreSQL single searches...")
            postgres_single_times = self.test_postgres_search(num_queries)
            postgres_single_stats = {
                'mean': statistics.mean(postgres_single_times),
                'median': statistics.median(postgres_single_times),
                'p95': np.percentile(postgres_single_times, 95),
                'p99': np.percentile(postgres_single_times, 99),
                'qps': 1.0 / statistics.mean(postgres_single_times)
            }
            print(f"  Mean: {postgres_single_stats['mean']:.4f}s")
            print(f"  QPS: {postgres_single_stats['qps']:.2f}")
        else:
            postgres_single_stats = None
            print("  PostgreSQL: Not available")
        
        print()
        
        # Batch search performance
        print("2. Batch Search Performance")
        print("-" * 30)
        
        if 'error' not in stats['qdrant']:
            print("Testing Qdrant batch searches...")
            qdrant_batch_times = self.test_qdrant_batch_search(num_batches, batch_size)
            qdrant_batch_stats = {
                'mean': statistics.mean(qdrant_batch_times),
                'median': statistics.median(qdrant_batch_times),
                'throughput': (batch_size * num_batches) / sum(qdrant_batch_times)
            }
            print(f"  Mean: {qdrant_batch_stats['mean']:.4f}s per batch")
            print(f"  Throughput: {qdrant_batch_stats['throughput']:.2f} queries/sec")
        else:
            qdrant_batch_stats = None
            print("  Qdrant: Not available")
        
        if 'error' not in stats['postgres']:
            print("Testing PostgreSQL batch searches...")
            postgres_batch_times = self.test_postgres_batch_search(num_batches, batch_size)
            postgres_batch_stats = {
                'mean': statistics.mean(postgres_batch_times),
                'median': statistics.median(postgres_batch_times),
                'throughput': (batch_size * num_batches) / sum(postgres_batch_times)
            }
            print(f"  Mean: {postgres_batch_stats['mean']:.4f}s per batch")
            print(f"  Throughput: {postgres_batch_stats['throughput']:.2f} queries/sec")
        else:
            postgres_batch_stats = None
            print("  PostgreSQL: Not available")
        
        print()
        
        # Concurrent search performance
        print("3. Concurrent Search Performance")
        print("-" * 30)
        
        if 'error' not in stats['qdrant'] and 'error' not in stats['postgres']:
            print("Testing concurrent searches...")
            qdrant_concurrent_times, postgres_concurrent_times = self.test_concurrent_searches(num_queries, 10)
            
            qdrant_concurrent_stats = {
                'mean': statistics.mean(qdrant_concurrent_times),
                'qps': 1.0 / statistics.mean(qdrant_concurrent_times)
            }
            postgres_concurrent_stats = {
                'mean': statistics.mean(postgres_concurrent_times),
                'qps': 1.0 / statistics.mean(postgres_concurrent_times)
            }
            
            print(f"  Qdrant: {qdrant_concurrent_stats['qps']:.2f} QPS")
            print(f"  PostgreSQL: {postgres_concurrent_stats['qps']:.2f} QPS")
        else:
            print("  Concurrent testing: Not available")
        
        print()
        
        # Summary
        print("4. Performance Summary")
        print("-" * 30)
        
        if qdrant_single_stats and postgres_single_stats:
            qdrant_qps = qdrant_single_stats['qps']
            postgres_qps = postgres_single_stats['qps']
            
            if qdrant_qps > postgres_qps:
                winner = "Qdrant"
                ratio = qdrant_qps / postgres_qps
            else:
                winner = "PostgreSQL"
                ratio = postgres_qps / qdrant_qps
            
            print(f"Single Search Winner: {winner}")
            print(f"Performance ratio: {ratio:.2f}x")
            print(f"Qdrant: {qdrant_qps:.2f} QPS")
            print(f"PostgreSQL: {postgres_qps:.2f} QPS")
        
        # Save results
        results = {
            'test_parameters': {
                'num_queries': num_queries,
                'num_batches': num_batches,
                'batch_size': batch_size,
                'vector_dimension': self.vector_dim
            },
            'database_stats': stats,
            'single_search': {
                'qdrant': qdrant_single_stats,
                'postgres': postgres_single_stats
            },
            'batch_search': {
                'qdrant': qdrant_batch_stats,
                'postgres': postgres_batch_stats
            }
        }
        
        with open('database_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: database_comparison_results.json")


def main():
    parser = argparse.ArgumentParser(description="Compare Qdrant and PostgreSQL vector databases")
    parser.add_argument("--queries", type=int, default=100, help="Number of single queries to test")
    parser.add_argument("--batches", type=int, default=10, help="Number of batch queries to test")
    parser.add_argument("--batch-size", type=int, default=10, help="Size of each batch")
    
    args = parser.parse_args()
    
    comparison = DatabaseComparison()
    
    try:
        comparison.run_comparison(
            num_queries=args.queries,
            num_batches=args.batches,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
    except Exception as e:
        print(f"Error during comparison: {e}")
        raise
    finally:
        comparison.postgres_conn.close()


if __name__ == "__main__":
    main()

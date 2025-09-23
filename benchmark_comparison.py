#!/usr/bin/env python3
"""
Vector Database Performance Comparison
Compares Qdrant vs PostgreSQL with pgvector across different scenarios
"""

import time
import json
import argparse
import statistics
from datetime import datetime
from benchmark_reads import ReadBenchmark
from benchmark_writes import WriteBenchmark
from compare_databases import DatabaseComparison
import psycopg2
from qdrant_client import QdrantClient


class DatabaseComparisonBenchmark:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, 
                 postgres_host="localhost", postgres_port=5432,
                 postgres_user="postgres", postgres_password="postgres",
                 postgres_db="vectordb"):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.postgres_config = {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db
        }
        
    def get_collection_info(self):
        """Get information about existing collections"""
        qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        collections = qdrant_client.get_collections()
        
        collection_info = {}
        for collection in collections.collections:
            try:
                info = qdrant_client.get_collection(collection.name)
                collection_info[collection.name] = {
                    "points_count": info.points_count,
                    "vector_dim": info.config.params.vectors.size,
                    "status": info.status
                }
            except Exception as e:
                print(f"Warning: Could not get info for collection {collection.name}: {e}")
                
        return collection_info
    
    def get_postgres_info(self):
        """Get information about PostgreSQL tables"""
        try:
            conn = psycopg2.connect(**self.postgres_config)
            with conn.cursor() as cur:
                # First get the count
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                count_result = cur.fetchone()
                
                # Then get vector dimension from a sample
                cur.execute("""
                    SELECT array_length(embedding::float[], 1) as vector_dim
                    FROM vector_embeddings 
                    LIMIT 1;
                """)
                dim_result = cur.fetchone()
                
                if count_result and dim_result:
                    return {
                        "points_count": count_result[0],
                        "vector_dim": dim_result[0] if dim_result[0] else 0
                    }
        except Exception as e:
            print(f"Warning: Could not get PostgreSQL info: {e}")
        return {"points_count": 0, "vector_dim": 0}
    
    def run_qdrant_benchmark(self, collection_name, iterations=100):
        """Run Qdrant benchmark on existing collection"""
        print(f"\n{'='*60}")
        print(f"QDRANT BENCHMARK - Collection: {collection_name}")
        print(f"{'='*60}")
        
        # Read benchmark
        read_benchmark = ReadBenchmark(
            host=self.qdrant_host,
            port=self.qdrant_port,
            collection_name=collection_name
        )
        
        print("Running read benchmark...")
        read_benchmark.run_benchmark_suite(iterations=iterations)
        
        # Write benchmark (use a separate collection for writes)
        write_collection = f"{collection_name}_write_test"
        write_benchmark = WriteBenchmark(
            host=self.qdrant_host,
            port=self.qdrant_port,
            collection_name=write_collection
        )
        
        print("Running write benchmark...")
        write_benchmark.create_test_collection()
        write_benchmark.run_benchmark_suite(iterations=iterations)
        write_benchmark.cleanup()
        
        return {
            "read": read_benchmark.results,
            "write": write_benchmark.results
        }
    
    def run_postgres_benchmark(self, iterations=100):
        """Run PostgreSQL benchmark"""
        print(f"\n{'='*60}")
        print(f"POSTGRESQL BENCHMARK")
        print(f"{'='*60}")
        
        # Use the existing compare_databases functionality
        db_comparison = DatabaseComparison()
        
        print("Running PostgreSQL benchmark...")
        postgres_results = db_comparison.run_comparison(num_queries=iterations)
        
        return postgres_results
    
    def calculate_performance_ratios(self, qdrant_results, postgres_results):
        """Calculate performance ratios between databases"""
        ratios = {}
        
        # Read performance comparisons
        if "read" in qdrant_results and "read_performance" in postgres_results:
            qdrant_read = qdrant_results["read"]
            postgres_read = postgres_results["read_performance"]
            
            # Single search comparison
            if "single_search" in qdrant_read and "single_search" in postgres_read:
                qdrant_qps = qdrant_read["single_search"]["qps"]
                postgres_qps = postgres_read["single_search"]["qps"]
                if postgres_qps > 0:
                    ratios["single_search_qps"] = {
                        "qdrant_vs_postgres": qdrant_qps / postgres_qps,
                        "qdrant_qps": qdrant_qps,
                        "postgres_qps": postgres_qps
                    }
            
            # Batch search comparison
            if "batch_search" in qdrant_read and "batch_search" in postgres_read:
                qdrant_qps = qdrant_read["batch_search"]["qps"]
                postgres_qps = postgres_read["batch_search"]["qps"]
                if postgres_qps > 0:
                    ratios["batch_search_qps"] = {
                        "qdrant_vs_postgres": qdrant_qps / postgres_qps,
                        "qdrant_qps": qdrant_qps,
                        "postgres_qps": postgres_qps
                    }
        
        # Write performance comparisons
        if "write" in qdrant_results and "write_performance" in postgres_results:
            qdrant_write = qdrant_results["write"]
            postgres_write = postgres_results["write_performance"]
            
            # Single insert comparison
            if "single_insert" in qdrant_write and "single_insert" in postgres_write:
                qdrant_throughput = qdrant_write["single_insert"]["throughput"]
                postgres_throughput = postgres_write["single_insert"]["throughput"]
                if postgres_throughput > 0:
                    ratios["single_insert_throughput"] = {
                        "qdrant_vs_postgres": qdrant_throughput / postgres_throughput,
                        "qdrant_throughput": qdrant_throughput,
                        "postgres_throughput": postgres_throughput
                    }
            
            # Batch insert comparison
            if "batch_insert_100" in qdrant_write and "batch_insert_100" in postgres_write:
                qdrant_throughput = qdrant_write["batch_insert_100"]["throughput"]
                postgres_throughput = postgres_write["batch_insert_100"]["throughput"]
                if postgres_throughput > 0:
                    ratios["batch_insert_throughput"] = {
                        "qdrant_vs_postgres": qdrant_throughput / postgres_throughput,
                        "qdrant_throughput": qdrant_throughput,
                        "postgres_throughput": postgres_throughput
                    }
        
        return ratios
    
    def print_comparison_summary(self, qdrant_results, postgres_results, ratios, collection_info, postgres_info):
        """Print a comprehensive comparison summary"""
        print(f"\n{'='*80}")
        print("DATABASE PERFORMANCE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nDATABASE SIZES:")
        print(f"  Qdrant Collections:")
        for name, info in collection_info.items():
            print(f"    {name}: {info['points_count']:,} points, {info['vector_dim']}D vectors")
        
        print(f"  PostgreSQL:")
        print(f"    vector_embeddings: {postgres_info['points_count']:,} points, {postgres_info['vector_dim']}D vectors")
        
        print(f"\n{'='*80}")
        print("PERFORMANCE COMPARISON (Qdrant vs PostgreSQL)")
        print(f"{'='*80}")
        
        if "single_search_qps" in ratios:
            ratio = ratios["single_search_qps"]
            print(f"\n🔍 SINGLE SEARCH PERFORMANCE:")
            print(f"  Qdrant:     {ratio['qdrant_qps']:.1f} QPS")
            print(f"  PostgreSQL: {ratio['postgres_qps']:.1f} QPS")
            if ratio["qdrant_vs_postgres"] > 1:
                print(f"  🏆 Qdrant is {ratio['qdrant_vs_postgres']:.1f}x FASTER")
            else:
                print(f"  🏆 PostgreSQL is {1/ratio['qdrant_vs_postgres']:.1f}x FASTER")
        
        if "batch_search_qps" in ratios:
            ratio = ratios["batch_search_qps"]
            print(f"\n📦 BATCH SEARCH PERFORMANCE:")
            print(f"  Qdrant:     {ratio['qdrant_qps']:.1f} QPS")
            print(f"  PostgreSQL: {ratio['postgres_qps']:.1f} QPS")
            if ratio["qdrant_vs_postgres"] > 1:
                print(f"  🏆 Qdrant is {ratio['qdrant_vs_postgres']:.1f}x FASTER")
            else:
                print(f"  🏆 PostgreSQL is {1/ratio['qdrant_vs_postgres']:.1f}x FASTER")
        
        if "single_insert_throughput" in ratios:
            ratio = ratios["single_insert_throughput"]
            print(f"\n✏️  SINGLE INSERT PERFORMANCE:")
            print(f"  Qdrant:     {ratio['qdrant_throughput']:.1f} ops/sec")
            print(f"  PostgreSQL: {ratio['postgres_throughput']:.1f} ops/sec")
            if ratio["qdrant_vs_postgres"] > 1:
                print(f"  🏆 Qdrant is {ratio['qdrant_vs_postgres']:.1f}x FASTER")
            else:
                print(f"  🏆 PostgreSQL is {1/ratio['qdrant_vs_postgres']:.1f}x FASTER")
        
        if "batch_insert_throughput" in ratios:
            ratio = ratios["batch_insert_throughput"]
            print(f"\n📝 BATCH INSERT PERFORMANCE:")
            print(f"  Qdrant:     {ratio['qdrant_throughput']:.1f} points/sec")
            print(f"  PostgreSQL: {ratio['postgres_throughput']:.1f} points/sec")
            if ratio["qdrant_vs_postgres"] > 1:
                print(f"  🏆 Qdrant is {ratio['qdrant_vs_postgres']:.1f}x FASTER")
            else:
                print(f"  🏆 PostgreSQL is {1/ratio['qdrant_vs_postgres']:.1f}x FASTER")
        
        print(f"\n{'='*80}")
        print("DETAILED PERFORMANCE METRICS")
        print(f"{'='*80}")
        
        # Detailed Qdrant metrics
        if "read" in qdrant_results:
            print(f"\nQDRANT READ PERFORMANCE:")
            read_results = qdrant_results["read"]
            for test_name, metrics in read_results.items():
                if isinstance(metrics, dict) and "qps" in metrics:
                    print(f"  {test_name.replace('_', ' ').title()}: {metrics['qps']:.1f} QPS")
        
        if "write" in qdrant_results:
            print(f"\nQDRANT WRITE PERFORMANCE:")
            write_results = qdrant_results["write"]
            for test_name, metrics in write_results.items():
                if isinstance(metrics, dict) and "throughput" in metrics:
                    print(f"  {test_name.replace('_', ' ').title()}: {metrics['throughput']:.1f} ops/sec")
        
        # Detailed PostgreSQL metrics
        if "read_performance" in postgres_results:
            print(f"\nPOSTGRESQL READ PERFORMANCE:")
            read_results = postgres_results["read_performance"]
            for test_name, metrics in read_results.items():
                if isinstance(metrics, dict) and "qps" in metrics:
                    print(f"  {test_name.replace('_', ' ').title()}: {metrics['qps']:.1f} QPS")
        
        if "write_performance" in postgres_results:
            print(f"\nPOSTGRESQL WRITE PERFORMANCE:")
            write_results = postgres_results["write_performance"]
            for test_name, metrics in write_results.items():
                if isinstance(metrics, dict) and "throughput" in metrics:
                    print(f"  {test_name.replace('_', ' ').title()}: {metrics['throughput']:.1f} ops/sec")
    
    def run_comparison_benchmark(self, iterations=100, qdrant_collection=None):
        """Run comprehensive comparison benchmark"""
        print("Vector Database Performance Comparison")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Iterations: {iterations}")
        
        # Get collection information
        collection_info = self.get_collection_info()
        postgres_info = self.get_postgres_info()
        
        if not collection_info:
            print("Error: No Qdrant collections found. Please populate Qdrant first.")
            return
        
        # Use specified collection or the largest one
        if qdrant_collection and qdrant_collection in collection_info:
            selected_collection = qdrant_collection
        else:
            selected_collection = max(collection_info.keys(), key=lambda k: collection_info[k]["points_count"])
        
        print(f"Selected Qdrant collection: {selected_collection}")
        print(f"Collection size: {collection_info[selected_collection]['points_count']:,} points")
        
        # Run benchmarks
        qdrant_results = self.run_qdrant_benchmark(selected_collection, iterations)
        postgres_results = self.run_postgres_benchmark(iterations)
        
        # Calculate performance ratios
        ratios = self.calculate_performance_ratios(qdrant_results, postgres_results)
        
        # Print comparison summary
        self.print_comparison_summary(qdrant_results, postgres_results, ratios, collection_info, postgres_info)
        
        # Save results
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "iterations": iterations,
                "qdrant_collection": selected_collection,
                "collection_info": collection_info,
                "postgres_info": postgres_info
            },
            "qdrant_results": qdrant_results,
            "postgres_results": postgres_results,
            "performance_ratios": ratios
        }
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        
        filename = f"results/database_comparison_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Compare Qdrant vs PostgreSQL performance")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations per test")
    parser.add_argument("--qdrant-collection", help="Specific Qdrant collection to test (default: largest)")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--postgres-db", default="vectordb", help="PostgreSQL database")
    
    args = parser.parse_args()
    
    benchmark = DatabaseComparisonBenchmark(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        postgres_db=args.postgres_db
    )
    
    try:
        results = benchmark.run_comparison_benchmark(
            iterations=args.iterations,
            qdrant_collection=args.qdrant_collection
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise


if __name__ == "__main__":
    main()

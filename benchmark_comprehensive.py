#!/usr/bin/env python3
"""
Comprehensive Vector Database Benchmark
Runs both read and write benchmarks with system monitoring.
"""

import time
import psutil
import os
import json
import argparse
from datetime import datetime
from benchmark_reads import ReadBenchmark
from benchmark_writes import WriteBenchmark


class ComprehensiveBenchmark:
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 read_collection: str = "test_vectors", 
                 write_collection: str = "write_test_vectors"):
        self.host = host
        self.port = port
        self.read_collection = read_collection
        self.write_collection = write_collection
        self.system_stats = {}
        
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
    
    def monitor_system_during_test(self, duration: int = 60):
        """Monitor system resources during test"""
        print(f"Monitoring system for {duration} seconds...")
        
        cpu_usage = []
        memory_usage = []
        disk_io = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=1))
            memory_usage.append(psutil.virtual_memory().percent)
            disk_io.append(psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {})
            time.sleep(1)
        
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
    
    def run_read_benchmark(self, iterations: int = 100):
        """Run read performance benchmark"""
        print("\n" + "="*60)
        print("RUNNING READ BENCHMARK")
        print("="*60)
        
        read_benchmark = ReadBenchmark(
            host=self.host,
            port=self.port,
            collection_name=self.read_collection
        )
        
        start_time = time.time()
        read_benchmark.run_benchmark_suite(iterations=iterations)
        end_time = time.time()
        
        return {
            "results": read_benchmark.results,
            "duration": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_write_benchmark(self, iterations: int = 100, use_existing_collections: bool = True):
        """Run write performance benchmark"""
        print("\n" + "="*60)
        print("RUNNING WRITE BENCHMARK")
        print("="*60)
        
        write_benchmark = WriteBenchmark(
            host=self.host,
            port=self.port,
            collection_name=self.write_collection
        )
        
        if use_existing_collections:
            # Check if write collection exists, if not create it
            try:
                collections = write_benchmark.client.get_collections()
                if not any(col.name == self.write_collection for col in collections.collections):
                    print(f"Write collection '{self.write_collection}' doesn't exist. Creating it...")
                    write_benchmark.create_test_collection()
                else:
                    print(f"Using existing write collection '{self.write_collection}'")
            except Exception as e:
                print(f"Error checking write collection: {e}")
                write_benchmark.create_test_collection()
        else:
            write_benchmark.create_test_collection()
        
        start_time = time.time()
        write_benchmark.run_benchmark_suite(iterations=iterations)
        end_time = time.time()
        
        # Clean up write test collection only if we created it
        if not use_existing_collections:
            write_benchmark.cleanup()
        
        return {
            "results": write_benchmark.results,
            "duration": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def run_load_test(self, duration: int = 120):
        """Run load test for specified duration"""
        print(f"\n" + "="*60)
        print(f"RUNNING LOAD TEST ({duration} seconds)")
        print("="*60)
        
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        read_benchmark = ReadBenchmark(
            host=self.host,
            port=self.port,
            collection_name=self.read_collection
        )
        
        def continuous_reads():
            """Continuously perform read operations"""
            while not stop_event.is_set():
                try:
                    read_benchmark.benchmark_single_search(limit=10)
                except Exception as e:
                    print(f"Error in continuous reads: {e}")
        
        def continuous_writes():
            """Continuously perform write operations"""
            write_benchmark = WriteBenchmark(
                host=self.host,
                port=self.port,
                collection_name=f"load_test_{int(time.time())}"
            )
            write_benchmark.create_test_collection()
            
            try:
                while not stop_event.is_set():
                    write_benchmark.benchmark_batch_insert(batch_size=100)
            except Exception as e:
                print(f"Error in continuous writes: {e}")
            finally:
                write_benchmark.cleanup()
        
        stop_event = threading.Event()
        
        # Start monitoring
        system_stats = self.monitor_system_during_test(duration)
        
        # Start load test
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Start read and write threads
            read_future = executor.submit(continuous_reads)
            write_future = executor.submit(continuous_writes)
            
            # Wait for duration
            time.sleep(duration)
            
            # Stop all operations
            stop_event.set()
            
            # Wait for threads to finish
            read_future.result(timeout=10)
            write_future.result(timeout=10)
        
        return system_stats
    
    def run_comprehensive_benchmark(self, iterations: int = 100, load_test_duration: int = 120, use_existing_collections: bool = True):
        """Run comprehensive benchmark suite"""
        print("Starting Comprehensive Vector Database Benchmark")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Host: {self.host}:{self.port}")
        print(f"Read Collection: {self.read_collection}")
        print(f"Write Collection: {self.write_collection}")
        print(f"Iterations: {iterations}")
        print(f"Load Test Duration: {load_test_duration}s")
        print(f"Use Existing Collections: {use_existing_collections}")
        
        # Get initial system info
        system_info = self.get_system_info()
        print(f"\nSystem Info:")
        print(f"CPU Cores: {system_info['cpu_count']}")
        print(f"Memory: {system_info['memory_total'] / (1024**3):.2f} GB")
        print(f"Available Memory: {system_info['memory_available'] / (1024**3):.2f} GB")
        print(f"Disk Usage: {system_info['disk_usage']:.1f}%")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "host": self.host,
                "port": self.port,
                "iterations": iterations,
                "load_test_duration": load_test_duration,
                "system_info": system_info
            },
            "read_benchmark": None,
            "write_benchmark": None,
            "load_test": None
        }
        
        try:
            # Run read benchmark
            results["read_benchmark"] = self.run_read_benchmark(iterations)
            
            # Run write benchmark
            results["write_benchmark"] = self.run_write_benchmark(iterations, use_existing_collections)
            
            # Run load test
            results["load_test"] = self.run_load_test(load_test_duration)
            
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
            read_results = results["read_benchmark"]["results"]
            for operation, stats in read_results.items():
                print(f"  {operation}: {stats['mean']:.4f}s mean, {1.0/stats['mean']:.2f} QPS")
        
        if results["write_benchmark"]:
            print("\nWRITE PERFORMANCE:")
            write_results = results["write_benchmark"]["results"]
            for operation, stats in write_results.items():
                if 'batch_size' in stats:
                    throughput = stats['batch_size'] / stats['mean']
                    print(f"  {operation}: {stats['mean']:.4f}s mean, {throughput:.2f} points/sec")
                else:
                    print(f"  {operation}: {stats['mean']:.4f}s mean, {1.0/stats['mean']:.2f} ops/sec")
        
        if results["load_test"]:
            print("\nLOAD TEST RESULTS:")
            load_stats = results["load_test"]
            print(f"  CPU Usage: {load_stats['cpu_usage']['mean']:.1f}% mean, {load_stats['cpu_usage']['max']:.1f}% max")
            print(f"  Memory Usage: {load_stats['memory_usage']['mean']:.1f}% mean, {load_stats['memory_usage']['max']:.1f}% max")
    
    def save_results(self, results, filename: str = "comprehensive_benchmark_results.json"):
        """Save comprehensive results to JSON file"""
        # Convert numpy types to Python types for JSON serialization
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
        import os
        os.makedirs("results", exist_ok=True)
        
        # Ensure filename is in results directory
        if not filename.startswith("results/"):
            filename = f"results/{filename}"
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nComprehensive results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive vector database benchmark")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--read-collection", default="test_vectors", help="Read benchmark collection")
    parser.add_argument("--write-collection", default="write_test_vectors", help="Write benchmark collection")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations per test")
    parser.add_argument("--load-duration", type=int, default=120, help="Load test duration in seconds")
    parser.add_argument("--output", default="comprehensive_benchmark_results.json", help="Output file for results")
    parser.add_argument("--use-existing-collections", action="store_true", default=True, help="Use existing collections instead of creating new ones")
    parser.add_argument("--create-new-collections", action="store_true", help="Create new collections for testing (overrides --use-existing-collections)")
    
    args = parser.parse_args()
    
    benchmark = ComprehensiveBenchmark(
        host=args.host,
        port=args.port,
        read_collection=args.read_collection,
        write_collection=args.write_collection
    )
    
    # Determine whether to use existing collections
    use_existing = args.use_existing_collections and not args.create_new_collections
    
    try:
        results = benchmark.run_comprehensive_benchmark(
            iterations=args.iterations,
            load_test_duration=args.load_duration,
            use_existing_collections=use_existing
        )
        benchmark.save_results(results, args.output)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error during comprehensive benchmark: {e}")
        raise


if __name__ == "__main__":
    main()

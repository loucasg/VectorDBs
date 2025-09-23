#!/usr/bin/env python3
"""
Simple Vector Database Performance Comparison
Compares Qdrant vs PostgreSQL with pgvector
"""

import time
import json
import argparse
import statistics
from datetime import datetime
from qdrant_client import QdrantClient
import psycopg2
import numpy as np


class SimpleComparison:
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
        
    def get_vector_dimension(self):
        """Get vector dimension from Qdrant collection"""
        try:
            collections = self.qdrant_client.get_collections()
            if collections.collections:
                collection_info = self.qdrant_client.get_collection(collections.collections[0].name)
                return collection_info.config.params.vectors.size
        except Exception as e:
            print(f"Warning: Could not get vector dimension from Qdrant: {e}")
        return 1024  # Default fallback
    
    def generate_query_vector(self, dim):
        """Generate a random normalized query vector"""
        vector = np.random.normal(0, 1, dim)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
    
    def test_qdrant_search(self, collection_name, num_queries=100):
        """Test Qdrant search performance"""
        times = []
        
        for _ in range(num_queries):
            query_vector = self.generate_query_vector(self.get_vector_dimension())
            
            start_time = time.time()
            try:
                results = self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=10
                )
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"Qdrant search error: {e}")
                return None
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "qps": num_queries / sum(times)
        }
    
    def test_postgres_search(self, num_queries=100):
        """Test PostgreSQL search performance"""
        times = []
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            with conn.cursor() as cur:
                for _ in range(num_queries):
                    query_vector = self.generate_query_vector(self.get_vector_dimension())
                    
                    start_time = time.time()
                    cur.execute("""
                        SELECT vector_id, text_content, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM vector_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT 10;
                    """, (query_vector, query_vector))
                    results = cur.fetchall()
                    end_time = time.time()
                    times.append(end_time - start_time)
        except Exception as e:
            print(f"PostgreSQL search error: {e}")
            return None
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "qps": num_queries / sum(times)
        }
    
    def test_qdrant_insert(self, collection_name, num_inserts=100):
        """Test Qdrant insert performance"""
        times = []
        
        try:
            # Create collection if it doesn't exist
            try:
                self.qdrant_client.get_collection(collection_name)
            except:
                from qdrant_client.models import VectorParams, Distance
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.get_vector_dimension(),
                        distance=Distance.COSINE
                    )
                )
            
            for i in range(num_inserts):
                vector = self.generate_query_vector(self.get_vector_dimension())
                point = {
                    "id": 10000 + i,  # Simple unique ID
                    "vector": vector,
                    "payload": {
                        "text": f"Test point {i}",
                        "metadata": {"test": True}
                    }
                }
                
                start_time = time.time()
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
                end_time = time.time()
                times.append(end_time - start_time)
        except Exception as e:
            print(f"Qdrant insert error: {e}")
            return None
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "ops_per_sec": num_inserts / sum(times)
        }
    
    def test_postgres_insert(self, num_inserts=100):
        """Test PostgreSQL insert performance"""
        times = []
        
        try:
            conn = psycopg2.connect(**self.postgres_config)
            with conn.cursor() as cur:
                for i in range(num_inserts):
                    vector = self.generate_query_vector(self.get_vector_dimension())
                    unique_id = 10000 + i  # Simple unique ID
                    
                    start_time = time.time()
                    cur.execute("""
                        INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
                        VALUES (%s, %s::vector, %s, %s);
                    """, (unique_id, vector, f"Test point {i}", '{"test": true}'))
                    conn.commit()
                    end_time = time.time()
                    times.append(end_time - start_time)
        except Exception as e:
            print(f"PostgreSQL insert error: {e}")
            return None
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "ops_per_sec": num_inserts / sum(times)
        }
    
    def run_comparison(self, qdrant_collection, num_queries=100, num_inserts=100):
        """Run comprehensive comparison"""
        print("Vector Database Performance Comparison")
        print("="*60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Qdrant Collection: {qdrant_collection}")
        print(f"Queries: {num_queries}, Inserts: {num_inserts}")
        
        # Get collection info
        try:
            collection_info = self.qdrant_client.get_collection(qdrant_collection)
            print(f"Qdrant Points: {collection_info.points_count:,}")
            print(f"Vector Dimension: {collection_info.config.params.vectors.size}")
        except Exception as e:
            print(f"Could not get Qdrant collection info: {e}")
        
        # Get PostgreSQL info
        try:
            conn = psycopg2.connect(**self.postgres_config)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                count = cur.fetchone()[0]
                print(f"PostgreSQL Points: {count:,}")
        except Exception as e:
            print(f"Could not get PostgreSQL info: {e}")
        
        print("\n" + "="*60)
        print("RUNNING BENCHMARKS")
        print("="*60)
        
        # Test searches
        print("\n🔍 Testing Search Performance...")
        qdrant_search = self.test_qdrant_search(qdrant_collection, num_queries)
        postgres_search = self.test_postgres_search(num_queries)
        
        # Test inserts
        print("\n✏️  Testing Insert Performance...")
        qdrant_insert = self.test_qdrant_insert(f"{qdrant_collection}_test", num_inserts)
        postgres_insert = self.test_postgres_insert(num_inserts)
        
        # Print results
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON RESULTS")
        print("="*80)
        
        if qdrant_search and postgres_search:
            print(f"\n🔍 SEARCH PERFORMANCE:")
            print(f"  Qdrant:     {qdrant_search['qps']:.1f} QPS (mean: {qdrant_search['mean']:.4f}s)")
            print(f"  PostgreSQL: {postgres_search['qps']:.1f} QPS (mean: {postgres_search['mean']:.4f}s)")
            
            if qdrant_search['qps'] > postgres_search['qps']:
                ratio = qdrant_search['qps'] / postgres_search['qps']
                print(f"  🏆 Qdrant is {ratio:.1f}x FASTER for searches")
            else:
                ratio = postgres_search['qps'] / qdrant_search['qps']
                print(f"  🏆 PostgreSQL is {ratio:.1f}x FASTER for searches")
        else:
            print(f"\n🔍 SEARCH PERFORMANCE: Could not complete tests")
        
        if qdrant_insert and postgres_insert:
            print(f"\n✏️  INSERT PERFORMANCE:")
            print(f"  Qdrant:     {qdrant_insert['ops_per_sec']:.1f} ops/sec (mean: {qdrant_insert['mean']:.4f}s)")
            print(f"  PostgreSQL: {postgres_insert['ops_per_sec']:.1f} ops/sec (mean: {postgres_insert['mean']:.4f}s)")
            
            if qdrant_insert['ops_per_sec'] > postgres_insert['ops_per_sec']:
                ratio = qdrant_insert['ops_per_sec'] / postgres_insert['ops_per_sec']
                print(f"  🏆 Qdrant is {ratio:.1f}x FASTER for inserts")
            else:
                ratio = postgres_insert['ops_per_sec'] / qdrant_insert['ops_per_sec']
                print(f"  🏆 PostgreSQL is {ratio:.1f}x FASTER for inserts")
        else:
            print(f"\n✏️  INSERT PERFORMANCE: Could not complete tests")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "qdrant_collection": qdrant_collection,
            "num_queries": num_queries,
            "num_inserts": num_inserts,
            "qdrant_search": qdrant_search,
            "postgres_search": postgres_search,
            "qdrant_insert": qdrant_insert,
            "postgres_insert": postgres_insert
        }
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        
        filename = f"results/simple_comparison_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Simple vector database comparison")
    parser.add_argument("--qdrant-collection", default="test_vectors", help="Qdrant collection to test")
    parser.add_argument("--queries", type=int, default=100, help="Number of search queries")
    parser.add_argument("--inserts", type=int, default=100, help="Number of insert operations")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--postgres-db", default="vectordb", help="PostgreSQL database")
    
    args = parser.parse_args()
    
    comparison = SimpleComparison(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password,
        postgres_db=args.postgres_db
    )
    
    try:
        results = comparison.run_comparison(
            qdrant_collection=args.qdrant_collection,
            num_queries=args.queries,
            num_inserts=args.inserts
        )
    except KeyboardInterrupt:
        print("\nComparison interrupted by user")
    except Exception as e:
        print(f"Error during comparison: {e}")
        raise


if __name__ == "__main__":
    main()

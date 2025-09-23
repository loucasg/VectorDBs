#!/usr/bin/env python3
"""
Database Reset Script
Drops existing collections and creates fresh empty ones for both Qdrant and PostgreSQL.
"""

import argparse
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import psycopg2
from psycopg2.extras import RealDictCursor


class DatabaseReset:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, 
                 postgres_host="localhost", postgres_port=5432,
                 postgres_db="vectordb", postgres_user="postgres", postgres_password="postgres"):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.postgres_conn = psycopg2.connect(
            host=postgres_host,
            port=postgres_port,
            database=postgres_db,
            user=postgres_user,
            password=postgres_password
        )
        self.vector_dim = 768
        
    def reset_qdrant(self, collections_to_reset=None):
        """Reset Qdrant collections"""
        print("🔄 Resetting Qdrant database...")
        
        try:
            # Get all collections
            collections = self.qdrant_client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if not existing_collections:
                print("  No collections found in Qdrant")
                return
            
            # Filter collections to reset
            if collections_to_reset:
                collections_to_drop = [col for col in existing_collections if col in collections_to_reset]
            else:
                collections_to_drop = existing_collections
            
            print(f"  Found {len(existing_collections)} collections: {existing_collections}")
            
            # Drop collections
            for collection_name in collections_to_drop:
                print(f"  Dropping collection: {collection_name}")
                self.qdrant_client.delete_collection(collection_name)
                print(f"    ✓ Dropped {collection_name}")
            
            print(f"  ✓ Qdrant reset complete - dropped {len(collections_to_drop)} collections")
            
        except Exception as e:
            print(f"  ✗ Error resetting Qdrant: {e}")
            raise
    
    def reset_postgres(self):
        """Reset PostgreSQL vector embeddings table"""
        print("🔄 Resetting PostgreSQL database...")
        
        try:
            with self.postgres_conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'vector_embeddings'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    print("  Dropping vector_embeddings table...")
                    cur.execute("DROP TABLE vector_embeddings CASCADE;")
                    print("    ✓ Dropped vector_embeddings table")
                else:
                    print("  No vector_embeddings table found")
                
                # Recreate the table with all indexes and functions
                print("  Creating fresh vector_embeddings table...")
                
                # Create table
                cur.execute("""
                    CREATE TABLE vector_embeddings (
                        id SERIAL PRIMARY KEY,
                        vector_id INTEGER UNIQUE NOT NULL,
                        embedding VECTOR(768),
                        text_content TEXT,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes
                print("  Creating indexes...")
                cur.execute("CREATE INDEX idx_vector_embeddings_vector_id ON vector_embeddings(vector_id);")
                cur.execute("CREATE INDEX idx_vector_embeddings_metadata ON vector_embeddings USING GIN(metadata);")
                cur.execute("CREATE INDEX idx_vector_embeddings_created_at ON vector_embeddings(created_at);")
                
                # Create HNSW index for vector similarity search
                cur.execute("""
                    CREATE INDEX idx_vector_embeddings_embedding_hnsw 
                    ON vector_embeddings USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                
                # Create trigger function
                cur.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql';
                """)
                
                # Create trigger
                cur.execute("""
                    CREATE TRIGGER update_vector_embeddings_updated_at 
                        BEFORE UPDATE ON vector_embeddings 
                        FOR EACH ROW 
                        EXECUTE FUNCTION update_updated_at_column();
                """)
                
                # Create search functions
                cur.execute("""
                    CREATE OR REPLACE FUNCTION search_similar_vectors(
                        query_embedding VECTOR(768),
                        match_limit INTEGER DEFAULT 10,
                        similarity_threshold FLOAT DEFAULT 0.0
                    )
                    RETURNS TABLE (
                        vector_id INTEGER,
                        text_content TEXT,
                        metadata JSONB,
                        similarity FLOAT
                    ) AS $$
                    BEGIN
                        RETURN QUERY
                        SELECT 
                            ve.vector_id,
                            ve.text_content,
                            ve.metadata,
                            1 - (ve.embedding <=> query_embedding) AS similarity
                        FROM vector_embeddings ve
                        WHERE 1 - (ve.embedding <=> query_embedding) > similarity_threshold
                        ORDER BY ve.embedding <=> query_embedding
                        LIMIT match_limit;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                
                cur.execute("""
                    CREATE OR REPLACE FUNCTION search_similar_vectors_batch(
                        query_embeddings VECTOR(768)[],
                        match_limit INTEGER DEFAULT 10,
                        similarity_threshold FLOAT DEFAULT 0.0
                    )
                    RETURNS TABLE (
                        query_index INTEGER,
                        vector_id INTEGER,
                        text_content TEXT,
                        metadata JSONB,
                        similarity FLOAT
                    ) AS $$
                    DECLARE
                        query_embedding VECTOR(768);
                        query_idx INTEGER;
                    BEGIN
                        FOR query_idx IN 1..array_length(query_embeddings, 1) LOOP
                            query_embedding := query_embeddings[query_idx];
                            
                            RETURN QUERY
                            SELECT 
                                query_idx,
                                ve.vector_id,
                                ve.text_content,
                                ve.metadata,
                                1 - (ve.embedding <=> query_embedding) AS similarity
                            FROM vector_embeddings ve
                            WHERE 1 - (ve.embedding <=> query_embedding) > similarity_threshold
                            ORDER BY ve.embedding <=> query_embedding
                            LIMIT match_limit;
                        END LOOP;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                
                cur.execute("""
                    CREATE OR REPLACE FUNCTION get_collection_stats()
                    RETURNS TABLE (
                        total_points BIGINT,
                        vector_dimensions INTEGER,
                        avg_metadata_size FLOAT,
                        created_at_range TEXT
                    ) AS $$
                    BEGIN
                        RETURN QUERY
                        SELECT 
                            COUNT(*)::BIGINT as total_points,
                            768 as vector_dimensions,
                            AVG(LENGTH(metadata::TEXT)::FLOAT) as avg_metadata_size,
                            CONCAT(
                                MIN(created_at)::TEXT, 
                                ' to ', 
                                MAX(created_at)::TEXT
                            ) as created_at_range
                        FROM vector_embeddings;
                    END;
                    $$ LANGUAGE plpgsql;
                """)
                
                # Commit all changes
                self.postgres_conn.commit()
                print("    ✓ Created vector_embeddings table with indexes and functions")
                
                # Verify table is empty
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                count = cur.fetchone()[0]
                print(f"    ✓ Table verified - {count} records")
                
            print("  ✓ PostgreSQL reset complete")
            
        except Exception as e:
            print(f"  ✗ Error resetting PostgreSQL: {e}")
            self.postgres_conn.rollback()
            raise
    
    def create_sample_collections(self, create_qdrant=True, create_postgres=True):
        """Create sample empty collections for testing"""
        print("📝 Creating sample collections...")
        
        if create_qdrant:
            try:
                print("  Creating Qdrant sample collections...")
                
                # Create main test collection
                self.qdrant_client.create_collection(
                    collection_name="test_vectors",
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE
                    )
                )
                print("    ✓ Created test_vectors collection")
                
                # Create small test collection
                self.qdrant_client.create_collection(
                    collection_name="test_vectors_small",
                    vectors_config=VectorParams(
                        size=self.vector_dim,
                        distance=Distance.COSINE
                    )
                )
                print("    ✓ Created test_vectors_small collection")
                
            except Exception as e:
                print(f"    ✗ Error creating Qdrant collections: {e}")
        
        if create_postgres:
            try:
                print("  PostgreSQL table already created during reset")
                print("    ✓ vector_embeddings table ready")
            except Exception as e:
                print(f"    ✗ Error with PostgreSQL: {e}")
    
    def verify_reset(self):
        """Verify that databases are properly reset"""
        print("🔍 Verifying reset...")
        
        # Check Qdrant
        try:
            collections = self.qdrant_client.get_collections()
            qdrant_collections = [col.name for col in collections.collections]
            print(f"  Qdrant collections: {qdrant_collections}")
        except Exception as e:
            print(f"  Qdrant verification failed: {e}")
        
        # Check PostgreSQL
        try:
            with self.postgres_conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM vector_embeddings;")
                postgres_count = cur.fetchone()[0]
                print(f"  PostgreSQL records: {postgres_count}")
        except Exception as e:
            print(f"  PostgreSQL verification failed: {e}")
    
    def reset_all(self, collections_to_reset=None, create_samples=True):
        """Reset both databases completely"""
        print("🚀 Starting complete database reset...")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Reset Qdrant
            self.reset_qdrant(collections_to_reset)
            print()
            
            # Reset PostgreSQL
            self.reset_postgres()
            print()
            
            # Create sample collections if requested
            if create_samples:
                self.create_sample_collections()
                print()
            
            # Verify reset
            self.verify_reset()
            print()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print("=" * 50)
            print("✅ Database reset completed successfully!")
            print(f"⏱️  Total time: {duration:.2f} seconds")
            print()
            print("Next steps:")
            print("  • Populate databases: python populate_qdrant.py --records 100000")
            print("  • Populate PostgreSQL: python populate_postgres.py --records 100000")
            print("  • Run benchmarks: python compare_databases.py --queries 50")
            print("  • Access web UI: python simple_ui.py --ui-port 5000")
            
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            raise
        finally:
            self.postgres_conn.close()


def main():
    parser = argparse.ArgumentParser(description="Reset vector databases")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--postgres-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--postgres-db", default="vectordb", help="PostgreSQL database name")
    parser.add_argument("--postgres-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--postgres-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--collections", nargs="+", help="Specific Qdrant collections to reset (default: all)")
    parser.add_argument("--no-samples", action="store_true", help="Don't create sample collections")
    parser.add_argument("--qdrant-only", action="store_true", help="Reset only Qdrant")
    parser.add_argument("--postgres-only", action="store_true", help="Reset only PostgreSQL")
    
    args = parser.parse_args()
    
    # Create reset instance
    reset = DatabaseReset(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        postgres_host=args.postgres_host,
        postgres_port=args.postgres_port,
        postgres_db=args.postgres_db,
        postgres_user=args.postgres_user,
        postgres_password=args.postgres_password
    )
    
    try:
        if args.qdrant_only:
            reset.reset_qdrant(args.collections)
            if not args.no_samples:
                reset.create_sample_collections(create_qdrant=True, create_postgres=False)
        elif args.postgres_only:
            reset.reset_postgres()
            if not args.no_samples:
                reset.create_sample_collections(create_qdrant=False, create_postgres=True)
        else:
            reset.reset_all(args.collections, not args.no_samples)
            
    except KeyboardInterrupt:
        print("\n⚠️  Reset interrupted by user")
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        raise


if __name__ == "__main__":
    main()

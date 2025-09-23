-- PostgreSQL initialization script for vector database
-- This script sets up the database with pgvector extension and sample tables

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for vector embeddings
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    vector_id INTEGER UNIQUE NOT NULL,
    embedding VECTOR(768),  -- 768-dimensional vectors
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_vector_id ON vector_embeddings(vector_id);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_metadata ON vector_embeddings USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_created_at ON vector_embeddings(created_at);

-- Create a specialized vector index for similarity search
-- Using HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_embedding_hnsw 
ON vector_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_vector_embeddings_updated_at 
    BEFORE UPDATE ON vector_embeddings 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create a function for similarity search
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

-- Create a function for batch similarity search
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

-- Create a function to get collection statistics
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

-- Insert some sample data for testing
INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata) VALUES
(1, ARRAY[0.1, 0.2, 0.3, 0.4, 0.5]::VECTOR(768), 'Sample document 1', '{"category": "test", "source": "sample"}'),
(2, ARRAY[0.2, 0.3, 0.4, 0.5, 0.6]::VECTOR(768), 'Sample document 2', '{"category": "test", "source": "sample"}'),
(3, ARRAY[0.3, 0.4, 0.5, 0.6, 0.7]::VECTOR(768), 'Sample document 3', '{"category": "test", "source": "sample"}')
ON CONFLICT (vector_id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE vectordb TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

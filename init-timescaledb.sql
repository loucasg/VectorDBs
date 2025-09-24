-- TimescaleDB initialization script for vector database
-- This script sets up the database with both TimescaleDB and pgvector extensions

-- Enable TimescaleDB extension (must be first)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for vector embeddings with time-series capabilities
-- Note: For TimescaleDB, we need to include the partitioning column in the primary key
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL,
    vector_id INTEGER NOT NULL,
    embedding VECTOR(1024),  -- 1024-dimensional vectors
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, created_at)
);

-- Convert the table to a hypertable (TimescaleDB feature)
-- This partitions the data by time for better performance on time-series queries
SELECT create_hypertable('vector_embeddings', 'created_at', if_not_exists => TRUE);

-- Create indexes for better performance
-- Note: For hypertables, unique indexes must include the partitioning column
CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_embeddings_vector_id_time 
ON vector_embeddings(vector_id, created_at);
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
    query_embedding VECTOR(1024),
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

-- Create a function for time-range similarity search (TimescaleDB specific)
CREATE OR REPLACE FUNCTION search_similar_vectors_time_range(
    query_embedding VECTOR(1024),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    match_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    vector_id INTEGER,
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMP,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ve.vector_id,
        ve.text_content,
        ve.metadata,
        ve.created_at,
        1 - (ve.embedding <=> query_embedding) AS similarity
    FROM vector_embeddings ve
    WHERE ve.created_at BETWEEN start_time AND end_time
      AND 1 - (ve.embedding <=> query_embedding) > similarity_threshold
    ORDER BY ve.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Create a function for batch similarity search
CREATE OR REPLACE FUNCTION search_similar_vectors_batch(
    query_embeddings VECTOR(1024)[],
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
    query_embedding VECTOR(1024);
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

-- Create a function to get collection statistics (TimescaleDB enhanced)
CREATE OR REPLACE FUNCTION get_collection_stats()
RETURNS TABLE (
    total_points BIGINT,
    vector_dimensions INTEGER,
    avg_metadata_size FLOAT,
    created_at_range TEXT,
    hypertable_info TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_points,
        1024 as vector_dimensions,
        AVG(LENGTH(metadata::TEXT)::FLOAT) as avg_metadata_size,
        CONCAT(
            MIN(created_at)::TEXT, 
            ' to ', 
            MAX(created_at)::TEXT
        ) as created_at_range,
        'TimescaleDB hypertable enabled' as hypertable_info
    FROM vector_embeddings;
END;
$$ LANGUAGE plpgsql;

-- Create TimescaleDB-specific continuous aggregates for time-based analytics
-- This creates a materialized view that automatically aggregates data by hour
CREATE MATERIALIZED VIEW IF NOT EXISTS vector_embeddings_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', created_at) AS bucket,
    COUNT(*) as vectors_count,
    AVG(LENGTH(text_content)) as avg_text_length
FROM vector_embeddings
GROUP BY bucket
WITH NO DATA;

-- Enable automatic refresh of the continuous aggregate
SELECT add_continuous_aggregate_policy('vector_embeddings_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Insert some sample data for testing
INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata) VALUES
(1, ARRAY[0.1, 0.2, 0.3, 0.4, 0.5]::VECTOR(1024), 'TimescaleDB sample document 1', '{"category": "timeseries", "source": "sample"}'),
(2, ARRAY[0.2, 0.3, 0.4, 0.5, 0.6]::VECTOR(1024), 'TimescaleDB sample document 2', '{"category": "timeseries", "source": "sample"}'),
(3, ARRAY[0.3, 0.4, 0.5, 0.6, 0.7]::VECTOR(1024), 'TimescaleDB sample document 3', '{"category": "timeseries", "source": "sample"}')
ON CONFLICT (vector_id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE vectordb TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO postgres;

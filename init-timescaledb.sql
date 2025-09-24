-- TimescaleDB initialization script for vector database
-- This script sets up the database with both TimescaleDB and pgvector extensions

-- Enable TimescaleDB extension (must be first)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pgvectorscale extension for enhanced vector performance
CREATE EXTENSION IF NOT EXISTS vectorscale;

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

-- Configure memory for optimal index build performance
SET maintenance_work_mem = '2GB';

-- Create DiskANN index for high-performance vector similarity search
-- DiskANN provides better performance than HNSW for large datasets
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_embedding_diskann 
ON vector_embeddings USING diskann (embedding)
WITH (
    num_neighbors = 50,           -- Maximum number of neighbors per node
    search_list_size = 100,       -- Number of additional candidates during graph search
    max_alpha = 1.2,              -- Graph quality parameter during construction
    num_dimensions = 1024,        -- Vector dimensions
    storage_layout = memory_optimized  -- Optimize for memory usage
);

-- Create additional DiskANN index optimized for different distance metrics if needed
-- This creates a backup index with different parameters for comparison
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_embedding_diskann_l2 
ON vector_embeddings USING diskann (embedding vector_l2_ops)
WITH (
    num_neighbors = 30,
    search_list_size = 75,
    max_alpha = 1.0
);

-- Set query-time parameters for optimal performance
-- These can be adjusted based on accuracy vs speed requirements
SET vectorscale.query_rescore = 400;  -- Number of elements to rescore during query

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

-- Create an optimized function for similarity search using DiskANN
CREATE OR REPLACE FUNCTION search_similar_vectors(
    query_embedding VECTOR(1024),
    match_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0,
    use_diskann BOOLEAN DEFAULT TRUE
)
RETURNS TABLE (
    vector_id INTEGER,
    text_content TEXT,
    metadata JSONB,
    similarity FLOAT,
    distance FLOAT
) AS $$
BEGIN
    -- Set DiskANN query parameters for optimal performance
    IF use_diskann THEN
        -- Adjust query-time parameters based on requirements
        PERFORM set_config('vectorscale.query_rescore', '400', true);
    END IF;
    
    RETURN QUERY
    SELECT 
        ve.vector_id,
        ve.text_content,
        ve.metadata,
        1 - (ve.embedding <=> query_embedding) AS similarity,
        ve.embedding <=> query_embedding AS distance
    FROM vector_embeddings ve
    WHERE 1 - (ve.embedding <=> query_embedding) > similarity_threshold
    ORDER BY ve.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Create a function for time-range similarity search (TimescaleDB + DiskANN optimized)
CREATE OR REPLACE FUNCTION search_similar_vectors_time_range(
    query_embedding VECTOR(1024),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    match_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0,
    use_diskann BOOLEAN DEFAULT TRUE
)
RETURNS TABLE (
    vector_id INTEGER,
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ,
    similarity FLOAT,
    distance FLOAT
) AS $$
BEGIN
    -- Optimize DiskANN parameters for time-range queries
    IF use_diskann THEN
        PERFORM set_config('vectorscale.query_rescore', '300', true);
    END IF;
    
    RETURN QUERY
    SELECT 
        ve.vector_id,
        ve.text_content,
        ve.metadata,
        ve.created_at,
        1 - (ve.embedding <=> query_embedding) AS similarity,
        ve.embedding <=> query_embedding AS distance
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

-- Create a function to get collection statistics (TimescaleDB + pgvectorscale enhanced)
CREATE OR REPLACE FUNCTION get_collection_stats()
RETURNS TABLE (
    total_points BIGINT,
    vector_dimensions INTEGER,
    avg_metadata_size FLOAT,
    created_at_range TEXT,
    hypertable_info TEXT,
    diskann_indexes TEXT,
    index_sizes TEXT
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
        'TimescaleDB hypertable with pgvectorscale' as hypertable_info,
        'DiskANN indexes: cosine, l2' as diskann_indexes,
        pg_size_pretty(pg_total_relation_size('vector_embeddings')) as index_sizes
    FROM vector_embeddings;
END;
$$ LANGUAGE plpgsql;

-- Create a function to optimize DiskANN query performance
CREATE OR REPLACE FUNCTION optimize_diskann_query(
    rescore_factor INTEGER DEFAULT 400,
    enable_parallel BOOLEAN DEFAULT TRUE
)
RETURNS TEXT AS $$
BEGIN
    -- Set optimal DiskANN query parameters
    PERFORM set_config('vectorscale.query_rescore', rescore_factor::TEXT, false);
    
    -- Enable parallel query execution if requested
    IF enable_parallel THEN
        PERFORM set_config('max_parallel_workers_per_gather', '4', false);
        PERFORM set_config('enable_parallel_append', 'on', false);
    END IF;
    
    RETURN format('DiskANN optimized: rescore=%s, parallel=%s', 
                  rescore_factor, enable_parallel);
END;
$$ LANGUAGE plpgsql;

-- Create a function to get DiskANN index statistics
CREATE OR REPLACE FUNCTION get_diskann_index_stats()
RETURNS TABLE (
    index_name TEXT,
    index_size TEXT,
    table_name TEXT,
    index_type TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.indexname::TEXT,
        pg_size_pretty(pg_relation_size(i.indexname::regclass))::TEXT,
        i.tablename::TEXT,
        'DiskANN'::TEXT
    FROM pg_indexes i
    WHERE i.tablename = 'vector_embeddings' 
      AND i.indexdef LIKE '%diskann%';
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

-- Configure TimescaleDB + pgvectorscale optimization settings
-- These settings optimize performance for vector operations
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb,vectorscale';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Set default DiskANN query parameters for optimal performance
ALTER SYSTEM SET vectorscale.query_rescore = 400;

-- Reload configuration
SELECT pg_reload_conf();

-- Display initialization summary
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB + pgvectorscale initialization completed!';
    RAISE NOTICE 'Features enabled:';
    RAISE NOTICE '  ✓ TimescaleDB hypertables for time-series partitioning';
    RAISE NOTICE '  ✓ pgvector for vector operations';
    RAISE NOTICE '  ✓ pgvectorscale for enhanced DiskANN performance';
    RAISE NOTICE '  ✓ DiskANN indexes for fast similarity search';
    RAISE NOTICE '  ✓ Continuous aggregates for time-based analytics';
    RAISE NOTICE '  ✓ Optimized query functions';
    RAISE NOTICE 'Use search_similar_vectors() for optimal vector queries';
END $$;

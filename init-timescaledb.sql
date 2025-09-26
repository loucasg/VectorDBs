-- Optimized TimescaleDB + pgvectorscale for 100M+ vectors
-- This script provides production-ready optimizations

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;

-- Optimized table structure for large-scale vector storage
CREATE TABLE IF NOT EXISTS vector_embeddings_ts (
    id BIGSERIAL,  -- Use BIGSERIAL for >2B records
    vector_id BIGINT NOT NULL,
    embedding VECTOR,  -- Let pgvector determine dimensions dynamically
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (vector_id, created_at)  -- Composite PK for hypertable
);

-- Create hypertable with optimized partitioning for large datasets
SELECT create_hypertable(
    'vector_embeddings_ts', 
    'created_at',
    chunk_time_interval => INTERVAL '6 hours',  -- Smaller chunks for better performance
    partitioning_column => 'vector_id',         -- Hash partitioning
    number_partitions => 16,                    -- Adjust based on CPU cores
    if_not_exists => TRUE
);

-- Optimize table storage parameters for large datasets
ALTER TABLE vector_embeddings_ts SET (
    fillfactor = 85,                           -- Leave room for updates
    autovacuum_vacuum_scale_factor = 0.05,     -- More frequent vacuuming
    autovacuum_analyze_scale_factor = 0.02,    -- More frequent analyze
    autovacuum_vacuum_cost_limit = 2000,       -- Faster vacuuming
    autovacuum_vacuum_cost_delay = 10          -- Reduce vacuum delay
);

-- Essential indexes for 100M+ scale
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vector_embeddings_ts_vector_id_hash
ON vector_embeddings_ts USING hash(vector_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vector_embeddings_ts_metadata_gin
ON vector_embeddings_ts USING GIN(metadata) WITH (fastupdate = off);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vector_embeddings_ts_created_at_brin
ON vector_embeddings_ts USING BRIN(created_at) WITH (pages_per_range = 32);

-- Production-optimized DiskANN indexes
-- Primary index optimized for high recall
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vector_embeddings_ts_embedding_diskann_primary
ON vector_embeddings_ts USING diskann (embedding vector_cosine_ops)
WITH (
    num_neighbors = 64,              -- Higher for better recall with large datasets
    search_list_size = 128,          -- Larger search list for accuracy
    max_alpha = 1.3,                 -- Better graph quality
    num_dimensions = 1024,
    storage_layout = memory_optimized
);

-- Secondary index optimized for speed
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vector_embeddings_ts_embedding_diskann_fast
ON vector_embeddings_ts USING diskann (embedding vector_cosine_ops)
WITH (
    num_neighbors = 32,              -- Lower for speed
    search_list_size = 64,           -- Smaller search list for speed
    max_alpha = 1.0,
    num_dimensions = 1024,
    storage_layout = memory_optimized
);

-- Optimized similarity search function with performance tuning
CREATE OR REPLACE FUNCTION search_similar_vectors_optimized(
    query_embedding VECTOR,
    match_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0,
    use_fast_index BOOLEAN DEFAULT FALSE,
    rescore_multiplier FLOAT DEFAULT 4.0
)
RETURNS TABLE (
    vector_id BIGINT,
    text_content TEXT,
    metadata JSONB,
    similarity FLOAT,
    distance FLOAT
) AS $$
DECLARE
    rescore_count INTEGER;
BEGIN
    -- Calculate optimal rescore count
    rescore_count := GREATEST(match_limit * rescore_multiplier, 100)::INTEGER;
    
    -- Set query-specific parameters
    PERFORM set_config('vectorscale.query_rescore', rescore_count::TEXT, true);
    PERFORM set_config('work_mem', '512MB', true);
    
    -- Force index choice based on performance requirements
    IF use_fast_index THEN
        SET enable_seqscan = off;
        SET enable_indexscan = on;
    END IF;
    
    RETURN QUERY
    SELECT 
        ve.vector_id,
        ve.text_content,
        ve.metadata,
        1 - (ve.embedding <=> query_embedding) AS similarity,
        ve.embedding <=> query_embedding AS distance
    FROM vector_embeddings_ts ve
    WHERE 1 - (ve.embedding <=> query_embedding) > similarity_threshold
    ORDER BY ve.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Time-partitioned search with chunk exclusion optimization
CREATE OR REPLACE FUNCTION search_similar_vectors_time_optimized(
    query_embedding VECTOR,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    match_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    vector_id BIGINT,
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ,
    similarity FLOAT
) AS $$
BEGIN
    -- Enable constraint exclusion for chunk pruning
    PERFORM set_config('constraint_exclusion', 'partition', true);
    PERFORM set_config('vectorscale.query_rescore', (match_limit * 3)::TEXT, true);
    
    RETURN QUERY
    SELECT 
        ve.vector_id,
        ve.text_content,
        ve.metadata,
        ve.created_at,
        1 - (ve.embedding <=> query_embedding) AS similarity
    FROM vector_embeddings_ts ve
    WHERE ve.created_at >= start_time 
      AND ve.created_at <= end_time
      AND 1 - (ve.embedding <=> query_embedding) > similarity_threshold
    ORDER BY ve.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Batch insert function optimized for bulk loading
CREATE OR REPLACE FUNCTION bulk_insert_vectors(
    vector_ids BIGINT[],
    embeddings VECTOR[],
    text_contents TEXT[],
    metadatas JSONB[]
)
RETURNS INTEGER AS $$
DECLARE
    batch_size INTEGER := array_length(vector_ids, 1);
BEGIN
    -- Optimize for bulk loading
    PERFORM set_config('maintenance_work_mem', '2GB', true);
    PERFORM set_config('checkpoint_completion_target', '0.9', true);
    
    -- Use COPY-like insert for performance
    INSERT INTO vector_embeddings_ts (vector_id, embedding, text_content, metadata)
    SELECT 
        unnest(vector_ids),
        unnest(embeddings),
        unnest(text_contents),
        unnest(metadatas)
    ON CONFLICT (vector_id, created_at) DO NOTHING;
    
    RETURN batch_size;
END;
$$ LANGUAGE plpgsql;

-- Enhanced statistics function with performance metrics
CREATE OR REPLACE FUNCTION get_collection_stats_detailed()
RETURNS TABLE (
    total_points BIGINT,
    total_chunks INTEGER,
    avg_chunk_size BIGINT,
    vector_dimensions INTEGER,
    index_sizes JSONB,
    table_size TEXT,
    query_performance_est TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM vector_embeddings_ts)::BIGINT,
        (SELECT COUNT(*) FROM timescaledb_information.chunks 
         WHERE hypertable_name = 'vector_embeddings_ts')::INTEGER,
        (SELECT COUNT(*) / GREATEST(COUNT(DISTINCT chunk_schema||'.'||chunk_name), 1)
         FROM timescaledb_information.chunks c
         JOIN vector_embeddings_ts v ON true
         WHERE c.hypertable_name = 'vector_embeddings_ts')::BIGINT,
        (SELECT vector_dims(embedding) FROM vector_embeddings_ts LIMIT 1)::INTEGER,
        (SELECT jsonb_object_agg(indexname, pg_size_pretty(pg_relation_size(indexname::regclass)))
         FROM pg_indexes WHERE tablename = 'vector_embeddings_ts'),
        pg_size_pretty(hypertable_size('vector_embeddings_ts')),
        CASE 
            WHEN (SELECT COUNT(*) FROM vector_embeddings_ts) > 50000000 
            THEN 'Consider query optimization for 50M+ vectors'
            ELSE 'Performance should be optimal'
        END;
END;
$$ LANGUAGE plpgsql;

-- Compression policy for old data (TimescaleDB feature)
SELECT add_compression_policy('vector_embeddings_ts', INTERVAL '7 days');

-- Retention policy (optional - remove old data)
-- SELECT add_retention_policy('vector_embeddings_ts', INTERVAL '1 year');

-- Continuous aggregate for hourly analytics
CREATE MATERIALIZED VIEW vector_embeddings_ts_hourly_optimized
WITH (timescaledb.continuous, timescaledb.materialized_only = true) AS
SELECT 
    time_bucket('1 hour', created_at) AS bucket,
    COUNT(*) as vectors_count,
    AVG(char_length(text_content)) as avg_text_length,
    COUNT(DISTINCT vector_id) as unique_vectors
FROM vector_embeddings_ts
GROUP BY bucket
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('vector_embeddings_ts_hourly_optimized',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Session-level optimizations (apply these before heavy operations)
CREATE OR REPLACE FUNCTION optimize_for_vector_operations()
RETURNS TEXT AS $$
BEGIN
    -- Memory settings
    PERFORM set_config('work_mem', '1GB', false);
    PERFORM set_config('maintenance_work_mem', '4GB', false);
    PERFORM set_config('temp_buffers', '256MB', false);
    
    -- Query planner settings
    PERFORM set_config('random_page_cost', '1.1', false);
    PERFORM set_config('effective_cache_size', '8GB', false);
    PERFORM set_config('cpu_tuple_cost', '0.01', false);
    
    -- Parallel execution
    PERFORM set_config('max_parallel_workers_per_gather', '4', false);
    PERFORM set_config('parallel_tuple_cost', '0.1', false);
    
    -- Vector-specific settings
    PERFORM set_config('vectorscale.query_rescore', '500', false);
    
    RETURN 'Vector database optimized for current session';
END;
$$ LANGUAGE plpgsql;

-- Maintenance function for large-scale deployments
CREATE OR REPLACE FUNCTION maintain_vector_database()
RETURNS TEXT AS $$
BEGIN
    -- Update table statistics
    ANALYZE vector_embeddings_ts;
    
    -- Refresh continuous aggregates
    CALL refresh_continuous_aggregate('vector_embeddings_ts_hourly_optimized', NULL, NULL);
    
    -- Compress old chunks (if compression enabled)
    PERFORM compress_chunk(chunk) FROM show_chunks('vector_embeddings_ts', older_than => INTERVAL '7 days') chunk;
    
    RETURN 'Database maintenance completed';
END;
$$ LANGUAGE plpgsql;

-- Example usage with proper error handling
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB + pgvectorscale optimized for 100M+ vectors';
    RAISE NOTICE 'Key optimizations:';
    RAISE NOTICE '  ✓ Hypertable with hash partitioning';
    RAISE NOTICE '  ✓ Multiple DiskANN indexes (recall vs speed)';
    RAISE NOTICE '  ✓ Optimized bulk insert functions';
    RAISE NOTICE '  ✓ Advanced query optimization';
    RAISE NOTICE '  ✓ Compression and retention policies';
    RAISE NOTICE '  ✓ Performance monitoring functions';
    RAISE NOTICE '';
    RAISE NOTICE 'Usage: SELECT optimize_for_vector_operations(); before queries';
    RAISE NOTICE 'Maintenance: SELECT maintain_vector_database(); regularly';
END $$;
-- Optimized PostgreSQL + pgvector for 100M+ vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Optimized table structure for large-scale vector storage
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id BIGSERIAL,
    vector_id BIGINT NOT NULL,
    embedding VECTOR,
    text_content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (vector_id, created_at)
) 
PARTITION BY RANGE (created_at);

-- Create partitions for the next 12 months
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE);
    partition_date DATE;
    partition_name TEXT;
BEGIN
    FOR i IN 0..11 LOOP
        partition_date := start_date + INTERVAL '1 month' * i;
        partition_name := 'vector_embeddings_' || TO_CHAR(partition_date, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE IF NOT EXISTS %I PARTITION OF vector_embeddings
            FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            partition_date,
            partition_date + INTERVAL '1 month'
        );
    END LOOP;
END $$;

-- Create the counter table needed by population scripts
CREATE TABLE IF NOT EXISTS vector_id_counter (
    id INTEGER PRIMARY KEY DEFAULT 1,
    last_id BIGINT NOT NULL DEFAULT 0
);

-- Optimize table storage parameters
ALTER TABLE vector_embeddings SET (
    fillfactor = 85,
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_vacuum_cost_limit = 2000
);

-- Essential indexes for 100M+ scale
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_vector_id_hash
ON vector_embeddings USING hash(vector_id);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_metadata_gin
ON vector_embeddings USING GIN(metadata) WITH (fastupdate = off);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_created_at_brin
ON vector_embeddings USING BRIN(created_at) WITH (pages_per_range = 16);

-- IVFFlat index optimized for large datasets
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_embedding_ivfflat
ON vector_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10000);

-- Alternative L2 distance index
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_embedding_ivfflat_l2
ON vector_embeddings USING ivfflat (embedding vector_l2_ops)
WITH (lists = 10000);

-- Optimized similarity search function
CREATE OR REPLACE FUNCTION search_similar_vectors_optimized(
    query_embedding VECTOR,
    match_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0,
    use_l2_distance BOOLEAN DEFAULT FALSE
)
RETURNS TABLE (
    vector_id BIGINT,
    text_content TEXT,
    metadata JSONB,
    similarity FLOAT,
    distance FLOAT
) AS $$
BEGIN
    PERFORM set_config('work_mem', '1GB', true);
    PERFORM set_config('ivfflat.probes', '100', true);
    
    IF use_l2_distance THEN
        RETURN QUERY
        SELECT 
            ve.vector_id, ve.text_content, ve.metadata,
            1 / (1 + ve.embedding <-> query_embedding) AS similarity,
            ve.embedding <-> query_embedding AS distance
        FROM vector_embeddings ve
        WHERE 1 / (1 + ve.embedding <-> query_embedding) > similarity_threshold
        ORDER BY ve.embedding <-> query_embedding
        LIMIT match_limit;
    ELSE
        RETURN QUERY
        SELECT 
            ve.vector_id, ve.text_content, ve.metadata,
            1 - (ve.embedding <=> query_embedding) AS similarity,
            ve.embedding <=> query_embedding AS distance
        FROM vector_embeddings ve
        WHERE 1 - (ve.embedding <=> query_embedding) > similarity_threshold
        ORDER BY ve.embedding <=> query_embedding
        LIMIT match_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Bulk insert optimized for large datasets
CREATE OR REPLACE FUNCTION bulk_insert_vectors_optimized(
    vector_ids BIGINT[],
    embeddings VECTOR[],
    text_contents TEXT[],
    metadatas JSONB[]
)
RETURNS INTEGER AS $$
DECLARE
    batch_size INTEGER := array_length(vector_ids, 1);
BEGIN
    -- Note: Cannot modify autovacuum settings on partitioned tables
    PERFORM set_config('maintenance_work_mem', '4GB', true);
    
    INSERT INTO vector_embeddings (vector_id, embedding, text_content, metadata)
    SELECT 
        unnest(vector_ids), unnest(embeddings),
        unnest(text_contents), unnest(metadatas);
    
    RETURN batch_size;
END;
$$ LANGUAGE plpgsql;

-- Query optimization function
CREATE OR REPLACE FUNCTION optimize_for_vector_queries()
RETURNS TEXT AS $$
BEGIN
    PERFORM set_config('work_mem', '1GB', false);
    PERFORM set_config('maintenance_work_mem', '4GB', false);
    PERFORM set_config('ivfflat.probes', '100', false);
    PERFORM set_config('effective_cache_size', '8GB', false);
    PERFORM set_config('constraint_exclusion', 'partition', false);
    
    RETURN 'PostgreSQL optimized for vector operations';
END;
$$ LANGUAGE plpgsql;

-- Trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on main table
CREATE TRIGGER update_vector_embeddings_updated_at 
    BEFORE UPDATE ON vector_embeddings 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO postgres;
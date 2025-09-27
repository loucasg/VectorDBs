-- TimescaleDB 17 + pgvector + vectorscale — 1024-dim
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;

CREATE TABLE IF NOT EXISTS vector_embeddings_ts (
  vector_id     BIGINT        NOT NULL,
  embedding     VECTOR(1024)  NOT NULL,
  text_content  TEXT,
  metadata      JSONB,
  created_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
  PRIMARY KEY (vector_id, created_at)
);

CREATE OR REPLACE FUNCTION trg_set_updated_at_ts() RETURNS trigger AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at_ts
BEFORE UPDATE ON vector_embeddings_ts
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at_ts();

SELECT create_hypertable('vector_embeddings_ts',
                         by_range('created_at'),
                         by_hash('vector_id', 16),
                         chunk_time_interval => INTERVAL '1 day',
                         if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS ve_ts_created_at_btree ON vector_embeddings_ts (created_at);
CREATE INDEX IF NOT EXISTS ve_ts_vector_id_btree  ON vector_embeddings_ts (vector_id);

CREATE INDEX IF NOT EXISTS ve_ts_embedding_vscale ON vector_embeddings_ts
USING vectorscale (embedding vector_l2_ops);

ALTER TABLE vector_embeddings_ts SET (
  timescaledb.compress = true,
  timescaledb.compress_segmentby = 'vector_id'
);

SELECT add_compression_policy('vector_embeddings_ts', INTERVAL '7 days') ON CONFLICT DO NOTHING;

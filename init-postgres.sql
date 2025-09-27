-- PostgreSQL 17 + pgvector (HNSW only) — 1024-dim
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vector_embeddings (
    vector_id     BIGINT        NOT NULL,
    embedding     VECTOR(1024)  NOT NULL,
    text_content  TEXT,
    metadata      JSONB,
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
    PRIMARY KEY (vector_id, created_at)
) PARTITION BY RANGE (created_at);

CREATE OR REPLACE FUNCTION trg_set_updated_at() RETURNS trigger AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at
BEFORE UPDATE ON vector_embeddings
FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

CREATE OR REPLACE FUNCTION create_month_partition(p_month_start DATE, p_hash_parts INT DEFAULT 16)
RETURNS VOID AS $$
DECLARE
  month_end DATE := (p_month_start + INTERVAL '1 month')::date;
  part_name TEXT := format('ve_%s', to_char(p_month_start, 'YYYYMM'));
  i INT;
  sub_name TEXT;
BEGIN
  EXECUTE format($f$
    CREATE TABLE IF NOT EXISTS %I PARTITION OF vector_embeddings
    FOR VALUES FROM (%L) TO (%L)
    PARTITION BY HASH (vector_id)
  $f$, part_name, p_month_start::timestamptz, month_end::timestamptz);

  FOR i IN 0..(p_hash_parts-1) LOOP
    sub_name := format('%s_h%s', part_name, i);
    EXECUTE format($f$
      CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
      FOR VALUES WITH (MODULUS %s, REMAINDER %s)
    $f$, sub_name, part_name, p_hash_parts, i);

    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (created_at);', sub_name||'_created_at_btree', sub_name);
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (vector_id);',   sub_name||'_vector_id_btree', sub_name);

    EXECUTE format(
      'CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=200);',
      sub_name||'_embedding_hnsw', sub_name
    );
  END LOOP;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
  start_month DATE := date_trunc('month', now())::date;
  k INT;
BEGIN
  FOR k IN 0..5 LOOP
    PERFORM create_month_partition(start_month + (k || ' months')::interval);
  END LOOP;
END $$;

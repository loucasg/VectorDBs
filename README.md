# Vector Database Testing Suite

Comprehensive benchmarking and testing suite for vector databases including Qdrant, PostgreSQL with pgvector, TimescaleDB with pgvector, Milvus, and Weaviate.

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start all databases
docker-compose up -d

# Wait for databases to be ready
docker-compose logs -f
```

### 2. Populate Databases
```bash
# Populate all databases with 100k records each
python populate_all.sh

# Or populate individually
python populate_qdrant.py --records 100000
python populate_postgres.py --records 100000
python populate_postgres_ts.py --records 100000
python populate_milvus.py --records 100000
python populate_weaviate.py --records 100000
```

### 3. Run Benchmarks
```bash
# Run all benchmarks
python benchmark.py

# Run specific benchmarks
python benchmark.py --read --write
python benchmark.py --postgres --postgres-ts
python benchmark.py --all-databases
```

### 4. Count Records
```bash
# Count records in all databases
python count_records.py

# Balance record counts across databases
python balance_databases.py --force
```

## Database Access Points

| Database | Type | Port | API | Description |
|----------|------|------|-----|-------------|
| **Qdrant** | Vector DB | 6333 | REST | High-performance vector database |
| **PostgreSQL** | SQL + Vector | 5432 | SQL | PostgreSQL with pgvector extension |
| **TimescaleDB** | Time-Series + Vector | 5433 | SQL | TimescaleDB with pgvector extension |
| **Milvus** | Vector DB | 19530 | gRPC | Open-source vector database |
| **Weaviate** | Vector DB | 8080 | REST/GraphQL | Open-source vector database |

## Scripts

### Main Scripts
- **`benchmark.py`** - Consolidated benchmark suite
- **`count_records.py`** - Count records across all databases
- **`balance_databases.py`** - Balance record counts for fair benchmarking
- **`populate_*.py`** - Populate individual databases
- **`reset_databases.py`** - Reset all databases to clean state

### Benchmark Usage
```bash
# Basic usage
python benchmark.py

# Specific tests
python benchmark.py --read --write
python benchmark.py --postgres --postgres-ts
python benchmark.py --milvus --weaviate
python benchmark.py --all-databases

# Custom parameters
python benchmark.py --iterations 50 --load-duration 60
```

### Command Line Options

#### Database Connection
- `--qdrant-host`, `--qdrant-port`: Qdrant settings
- `--postgres-host`, `--postgres-port`, `--postgres-user`, `--postgres-password`, `--postgres-db`: PostgreSQL settings
- `--postgres-ts-host`, `--postgres-ts-port`, `--postgres-ts-user`, `--postgres-ts-password`, `--postgres-ts-db`: TimescaleDB settings
- `--milvus-host`, `--milvus-port`: Milvus settings
- `--weaviate-host`, `--weaviate-port`: Weaviate settings

#### Test Selection
- `--all`: Run all benchmarks (default)
- `--read`: Read performance tests only
- `--write`: Write performance tests only
- `--postgres`: PostgreSQL benchmark only
- `--postgres-ts`: TimescaleDB benchmark only
- `--milvus`: Milvus benchmark only
- `--weaviate`: Weaviate benchmark only
- `--all-databases`: Run all database benchmarks
- `--comparison`: Database comparison tests
- `--load-test`: Load testing with system monitoring

#### Test Configuration
- `--iterations`: Number of iterations per test (default: 100)
- `--load-duration`: Load test duration in seconds (default: 120)
- `--read-collection`: Read benchmark collection (default: test_vectors)
- `--write-collection`: Write benchmark collection (default: test_vectors)

## Examples

```bash
# Quick test
python benchmark.py --iterations 10

# Read performance only
python benchmark.py --read --iterations 200

# All databases comparison
python benchmark.py --all-databases --iterations 100

# Load testing
python benchmark.py --load-test --load-duration 300

# Balance databases for fair comparison
python balance_databases.py --dry-run
python balance_databases.py --force
```

## Output

Results are saved to `results/` directory as JSON files with:
- Performance metrics (mean, median, P95, P99 times)
- Throughput (QPS, ops/sec)
- System monitoring (CPU, memory usage)
- Database comparison ratios

## Requirements

- Docker and Docker Compose
- Python 3.8+
- 8GB+ RAM recommended for large datasets

## Troubleshooting

```bash
# Check database status
docker-compose ps
docker-compose logs

# Reset all databases
python reset_databases.py

# Check record counts
python count_records.py --all-databases
```
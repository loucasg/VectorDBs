# Vector Database Testing Suite

Comprehensive benchmarking and testing suite for vector databases including Qdrant and PostgreSQL with pgvector.

## Features

- **Docker-based vector databases** - Qdrant and PostgreSQL with pgvector
- **10 million record population** with realistic chunk data
- **Comprehensive benchmarking** for read and write operations
- **Database comparison** - Side-by-side performance analysis
- **System monitoring** during tests
- **Concurrent operation testing**
- **Performance metrics** with detailed statistics
- **Consolidated benchmark suite** - Single script for all test types
- **Record counting utilities** - Count and compare records across databases
- **Web interface** - Custom UI for collection management and exploration

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.8+
- 8GB+ RAM recommended for 10M records

### 2. Web Interfaces

The setup includes a custom web interface:

- **Custom Web UI:** http://localhost:5000 - Enhanced interface with collection management
- **Qdrant API:** http://localhost:6333 - Direct API access for advanced users

### 3. Setup

```bash
# Clone or download the project files
cd /path/to/vector-db-test

# Install Python dependencies
pip install -r requirements.txt

# Start vector databases (Qdrant + PostgreSQL with pgvector)
docker-compose up -d

# Wait for databases to be ready (about 30 seconds)
docker-compose logs -f

# Start the custom web UI (optional)
python simple_ui.py --ui-port 5000
```

### 4. Reset Databases (Optional)

```bash
# Reset both databases to clean state
python reset_databases.py

# Reset only Qdrant
python reset_databases.py --qdrant-only

# Reset only PostgreSQL
python reset_databases.py --postgres-only

# Reset specific Qdrant collections
python reset_databases.py --collections test_vectors test_vectors_small
```

### 5. Populate Databases

```bash
# Populate Qdrant with 10 million records (this will take 30-60 minutes)
python populate_qdrant.py --records 10000000

# Populate PostgreSQL with test data
python populate_postgres.py --records 1000000

# Or start with smaller datasets for testing
python populate_qdrant.py --records 100000
python populate_postgres.py --records 10000

# Add more data to existing collections (won't drop existing data)
python populate_qdrant.py --records 50000 --collection test_vectors
python populate_postgres.py --records 5000

# Use different vector dimensions (PostgreSQL will recreate table if needed)
python populate_postgres.py --vector-dim 1024 --records 100000
python populate_qdrant.py --vector-dim 512 --records 50000
```

### 6. Run Benchmarks

#### Quick Start (Recommended)
```bash
# Run all benchmarks with default settings
python benchmark_all.py

# Run specific benchmarks only
python benchmark_all.py --read --write
python benchmark_all.py --comparison --load-test
```

#### All Functionality Available in benchmark_all.py
All individual benchmark functionality is now consolidated into the single `benchmark_all.py` script with flags for different test types.

### 7. Count Records

```bash
# Count records in test_vectors collection
python count_records.py

# Show all Qdrant collections
python count_records.py --all-collections

# Count specific collection
python count_records.py --collection my_collection
```

> **Note**: The `benchmark_all.py` script consolidates all benchmark functionality. See the [Benchmark Usage](#benchmark-usage) section for detailed usage instructions.

## Web Interfaces

### Custom Web UI
- **URL:** http://localhost:5000
- **Features:** 
  - Collection management (create, delete, explore)
  - Vector similarity search
  - Point browsing and exploration
  - Real-time collection statistics
- **Use for:** Interactive exploration and testing

### Qdrant API
- **URL:** http://localhost:6333
- **Features:** Direct REST API access
- **Use for:** Advanced users, programmatic access, debugging

## Scripts Overview

### Main Benchmark Script

- **`benchmark_all.py`** - **Consolidated benchmark suite** - Single script with flags for all test types
  - `--read` - Read performance tests (search, retrieve, scroll, concurrent)
  - `--write` - Write performance tests (insert, update, delete, batch operations)
  - `--postgres` - PostgreSQL performance tests
  - `--comparison` - Direct Qdrant vs PostgreSQL comparison
  - `--load-test` - Sustained performance testing with system monitoring
  - `--all` - Run all benchmarks (default)
  - See [Benchmark Usage](#benchmark-usage) section for detailed usage

### Core Scripts
- **`populate_qdrant.py`** - Populates Qdrant with test data (adds to existing collections, supports different vector dimensions)
- **`populate_postgres.py`** - Populates PostgreSQL with test data (adds to existing data, auto-adjusts table schema for different vector dimensions)
- **`reset_databases.py`** - Resets both databases to clean state
- **`count_records.py`** - Counts and compares records across both databases
- **`simple_ui.py`** - Custom web interface for collection management and exploration

### Configuration Files

- **`docker-compose.yml`** - Qdrant and PostgreSQL database setup
- **`init-postgres.sql`** - PostgreSQL initialization with pgvector
- **`requirements.txt`** - Python dependencies

## Benchmark Usage

The `benchmark_all.py` script consolidates all benchmark functionality into a single, easy-to-use tool with flags for different test types. This replaces the need for multiple separate benchmark scripts.

### Features

- **Read Benchmarks**: Single search, batch search, filtered search, ID retrieval, scrolling, concurrent searches
- **Write Benchmarks**: Single insert, batch insert (multiple sizes), concurrent inserts, updates, deletes
- **PostgreSQL Benchmarks**: Search and insert performance testing
- **Database Comparison**: Direct performance comparison between Qdrant and PostgreSQL
- **Load Testing**: Sustained performance testing with system monitoring
- **Flexible Execution**: Run all tests or select specific ones with flags

### Basic Usage

```bash
# Run all benchmarks (default)
python benchmark_all.py

# Run all benchmarks with custom parameters
python benchmark_all.py --iterations 50 --load-duration 60

# Run specific benchmarks only
python benchmark_all.py --read --write
python benchmark_all.py --postgres --comparison
python benchmark_all.py --load-test --load-duration 30
```

### Command Line Options

#### Database Connection
- `--qdrant-host`: Qdrant host (default: localhost)
- `--qdrant-port`: Qdrant port (default: 6333)
- `--postgres-host`: PostgreSQL host (default: localhost)
- `--postgres-port`: PostgreSQL port (default: 5432)
- `--postgres-user`: PostgreSQL user (default: postgres)
- `--postgres-password`: PostgreSQL password (default: postgres)
- `--postgres-db`: PostgreSQL database (default: vectordb)

#### Test Configuration
- `--read-collection`: Read benchmark collection (default: test_vectors)
- `--write-collection`: Write benchmark collection (default: test_vectors)
- `--iterations`: Number of iterations per test (default: 100)
- `--load-duration`: Load test duration in seconds (default: 120)

#### Test Selection Flags
- `--all`: Run all benchmarks (default if no specific flags)
- `--read`: Run read benchmark only
- `--write`: Run write benchmark only
- `--postgres`: Run PostgreSQL benchmark only
- `--comparison`: Run database comparison only
- `--load-test`: Run load test only

#### Output
- `--output`: Output file for results (default: comprehensive_benchmark_results.json)

### Examples

#### Quick Test
```bash
# Run a quick test with minimal iterations
python benchmark_all.py --iterations 10 --load-duration 30
```

#### Read Performance Only
```bash
# Test only read performance
python benchmark_all.py --read --iterations 200
```

#### Write Performance Only
```bash
# Test only write performance
python benchmark_all.py --write --iterations 50
```

#### Database Comparison
```bash
# Compare Qdrant vs PostgreSQL performance
python benchmark_all.py --comparison --iterations 100
```

#### Load Testing
```bash
# Run a 5-minute load test
python benchmark_all.py --load-test --load-duration 300
```

#### Custom Collections
```bash
# Test specific collections
python benchmark_all.py --read-collection my_vectors --write-collection my_test_vectors
```

#### Multiple Specific Tests
```bash
# Run read, write, and comparison tests
python benchmark_all.py --read --write --comparison --iterations 50
```

### Output

All results are saved to the `results/` directory as JSON files with comprehensive metrics including:

- **Performance Metrics**: Mean, median, P95, P99, min, max times
- **Throughput**: QPS (queries per second) and ops/sec (operations per second)
- **System Monitoring**: CPU and memory usage during load tests
- **Database Comparison**: Performance ratios between Qdrant and PostgreSQL

### Migration from Individual Scripts

The consolidated script replaces these individual scripts (now removed):
- `benchmark_reads.py` → `--read` flag
- `benchmark_writes.py` → `--write` flag
- `benchmark_comprehensive.py` → `--all` flag
- `compare_databases.py` → `--comparison` flag
- `simple_comparison.py` → `--comparison` flag
- `benchmark_comparison.py` → `--comparison` flag

### Benefits

1. **Single Entry Point**: One script for all benchmark types
2. **Flexible Execution**: Run specific tests or all tests
3. **Consistent Interface**: Same parameters and output format
4. **Easier Maintenance**: Single codebase to maintain
5. **Better Integration**: All tests work together seamlessly
6. **Comprehensive Results**: All metrics in one output file

### Requirements

- Python 3.7+
- Qdrant client library
- PostgreSQL with pgvector extension
- psycopg2-binary
- numpy
- psutil
- tqdm

### Notes

- The script automatically detects vector dimensions from existing collections
- Write benchmarks create temporary collections that are cleaned up automatically
- Load tests run concurrent read and write operations
- All results are saved to the `results/` directory
- The script handles errors gracefully and provides detailed output

## Record Counting

The `count_records.py` script provides a comprehensive way to count and compare records across both databases.

### Features

- **Count Qdrant collections** - Get record counts and metadata
- **Count PostgreSQL tables** - Get record counts and table size information
- **Compare databases** - Side-by-side comparison of record counts
- **Show all collections** - List all Qdrant collections with counts
- **Detailed metadata** - Vector dimensions, table sizes, indexed vectors

### Usage

```bash
# Count records in test_vectors collection
python count_records.py

# Show all Qdrant collections
python count_records.py --all-collections

# Count specific collection
python count_records.py --collection my_collection

# Use custom database settings
python count_records.py --qdrant-host localhost --postgres-host localhost
```

### Command Line Options

- `--collection`: Qdrant collection name to count (default: test_vectors)
- `--all-collections`: Show counts for all Qdrant collections
- `--qdrant-host`: Qdrant host (default: localhost)
- `--qdrant-port`: Qdrant port (default: 6333)
- `--postgres-host`: PostgreSQL host (default: localhost)
- `--postgres-port`: PostgreSQL port (default: 5432)
- `--postgres-user`: PostgreSQL user (default: postgres)
- `--postgres-password`: PostgreSQL password (default: postgres)
- `--postgres-db`: PostgreSQL database (default: vectordb)

### Output Information

**For Qdrant:**
- Collection name
- Total record count
- Vector dimension
- Distance metric
- Number of indexed vectors

**For PostgreSQL:**
- Table name
- Total record count
- Vector dimension
- Table size (total, data, index)

**Comparison:**
- Shows if databases have the same number of records
- Displays difference in record counts
- Clear winner indication

## Detailed Usage

### Database Reset

The reset script allows you to clean up and reinitialize both databases:

```bash
# Complete reset of both databases
python reset_databases.py

# Reset only Qdrant (keeps PostgreSQL data)
python reset_databases.py --qdrant-only

# Reset only PostgreSQL (keeps Qdrant data)
python reset_databases.py --postgres-only

# Reset specific Qdrant collections only
python reset_databases.py --collections test_vectors

# Reset without creating sample collections
python reset_databases.py --no-samples

# Custom database connections
python reset_databases.py \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --postgres-host localhost \
  --postgres-port 5432 \
  --postgres-db vectordb \
  --postgres-user postgres \
  --postgres-password postgres
```

**What the reset script does:**
- **Qdrant**: Drops all collections and recreates empty ones
- **PostgreSQL**: Drops and recreates the `vector_embeddings` table with all indexes and functions
- **Verification**: Confirms both databases are empty and ready for new data

### Database Population

The population script creates realistic chunk data with:
- 1024-dimensional vectors (normalized)
- Metadata including source, page, section, category
- Configurable batch sizes and worker threads

```bash
# Full options
python populate_qdrant.py \
  --host localhost \
  --port 6333 \
  --collection test_vectors \
  --records 10000000 \
  --workers 4 \
  --vector-dim 1024
```

### Read Benchmark

Tests various read operations:
- Single vector search
- Batch vector search
- Filtered search
- Retrieve by ID
- Collection scrolling
- Concurrent searches

```bash
# Full options
python benchmark_reads.py \
  --host localhost \
  --port 6333 \
  --collection test_vectors \
  --iterations 100 \
  --output read_results.json
```

### Write Benchmark

Tests various write operations:
- Single point insertion
- Batch insertions (different sizes)
- Concurrent insertions
- Update operations
- Delete operations

```bash
# Full options
python benchmark_writes.py \
  --host localhost \
  --port 6333 \
  --collection write_test_vectors \
  --iterations 100 \
  --output write_results.json \
  --cleanup
```

### Comprehensive Benchmark

Runs all tests with system monitoring:
- Read and write benchmarks
- Load testing
- CPU and memory monitoring
- Detailed performance reports

```bash
# Full options
python benchmark_comprehensive.py \
  --host localhost \
  --port 6333 \
  --read-collection test_vectors \
  --write-collection write_test_vectors \
  --iterations 100 \
  --load-duration 300 \
  --output comprehensive_results.json
```

## Performance Expectations

### Hardware Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores, SSD storage
- **For 10M records**: 32GB RAM, 16 CPU cores

### Expected Performance (on recommended hardware)

- **Population**: 5,000-15,000 records/second
- **Single search**: 1-5ms (95th percentile)
- **Batch search**: 10-50ms for 10 vectors
- **Concurrent searches**: 100-500 QPS
- **Batch inserts**: 1,000-5,000 points/second

## Monitoring and Debugging

### Docker Logs

```bash
# View Qdrant logs
docker-compose logs -f qdrant

# View all service logs
docker-compose logs -f
```

### Database Status

```bash
# Check Qdrant health
curl http://localhost:6333/health

# Get collection info
curl http://localhost:6333/collections/test_vectors
```

### System Resources

The comprehensive benchmark includes system monitoring:
- CPU usage (mean, max, min)
- Memory usage (mean, max, min)
- Disk I/O statistics

## Configuration

### Qdrant Configuration

The Docker setup uses default Qdrant configuration. For production use, consider:

- Adjusting memory limits
- Configuring persistence settings
- Setting up clustering
- Enabling authentication

### Python Configuration

Key parameters you can adjust:

- `vector_dim`: Vector dimensionality (default: 1024)
- `batch_size`: Batch size for operations (default: 1000)
- `max_workers`: Number of concurrent workers
- `iterations`: Number of test iterations

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or number of workers
2. **Connection Refused**: Ensure Qdrant is running (`docker-compose ps`)
3. **Slow Performance**: Check system resources and disk space
4. **Import Errors**: Install dependencies (`pip install -r requirements.txt`)

### Performance Tuning

1. **Increase batch sizes** for better throughput
2. **Adjust worker counts** based on CPU cores
3. **Use SSD storage** for better I/O performance
4. **Increase Docker memory limits** if needed

## Results Analysis

### Output Files

- `read_benchmark_results.json` - Read performance metrics
- `write_benchmark_results.json` - Write performance metrics
- `comprehensive_benchmark_results.json` - Complete benchmark results

### Key Metrics

- **Latency**: Mean, median, P95, P99 response times
- **Throughput**: Operations per second, points per second
- **Concurrency**: Performance under concurrent load
- **System**: CPU, memory, disk usage during tests

## Advanced Usage

### Custom Vector Dimensions

```bash
python populate_qdrant.py --vector-dim 1536 --records 1000000
```

### Custom Collections

```bash
python benchmark_reads.py --collection my_custom_collection
```

### Load Testing

```bash
python benchmark_comprehensive.py --load-duration 600  # 10 minutes
```

## Contributing

Feel free to extend the benchmark suite with:
- Additional vector databases (Pinecone, Weaviate, etc.)
- More complex query patterns
- Different data types and schemas
- Advanced filtering scenarios

## License

This project is provided as-is for testing and benchmarking purposes.

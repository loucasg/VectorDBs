# Vector Database Testing Suite

A comprehensive testing suite for vector databases using Qdrant, designed to populate databases with millions of records and benchmark performance.

## Features

- **Docker-based Qdrant setup** with persistent storage
- **10 million record population** with realistic chunk data
- **Comprehensive benchmarking** for read and write operations
- **System monitoring** during tests
- **Concurrent operation testing**
- **Performance metrics** with detailed statistics

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.8+
- 8GB+ RAM recommended for 10M records

### 2. Setup

```bash
# Clone or download the project files
cd /path/to/vector-db-test

# Install Python dependencies
pip install -r requirements.txt

# Start Qdrant database
docker-compose up -d

# Wait for Qdrant to be ready (about 30 seconds)
docker-compose logs -f qdrant
```

### 3. Populate Database

```bash
# Populate with 10 million records (this will take 30-60 minutes)
python populate_database.py --records 10000000

# Or start with a smaller dataset for testing
python populate_database.py --records 100000
```

### 4. Run Benchmarks

```bash
# Run read performance benchmark
python benchmark_reads.py

# Run write performance benchmark
python benchmark_writes.py

# Run comprehensive benchmark (includes load testing)
python benchmark_comprehensive.py
```

## Scripts Overview

### Core Scripts

- **`populate_database.py`** - Populates the database with test data
- **`benchmark_reads.py`** - Tests read performance (search, retrieve, scroll)
- **`benchmark_writes.py`** - Tests write performance (insert, update, delete)
- **`benchmark_comprehensive.py`** - Runs all benchmarks with system monitoring

### Configuration Files

- **`docker-compose.yml`** - Qdrant database setup
- **`requirements.txt`** - Python dependencies

## Detailed Usage

### Database Population

The population script creates realistic chunk data with:
- 768-dimensional vectors (normalized)
- Metadata including source, page, section, category
- Configurable batch sizes and worker threads

```bash
# Full options
python populate_database.py \
  --host localhost \
  --port 6333 \
  --collection test_vectors \
  --records 10000000 \
  --workers 4 \
  --vector-dim 768
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

- `vector_dim`: Vector dimensionality (default: 768)
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
python populate_database.py --vector-dim 1536 --records 1000000
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

#!/bin/bash

# Vector Database Testing Suite - Run All Tests
# This script runs the complete testing pipeline

set -e  # Exit on any error

echo "=========================================="
echo "Vector Database Testing Suite"
echo "=========================================="

# Configuration
HOST="localhost"
PORT="6333"
RECORDS=${1:-1000000}  # Default to 1M records if not specified
ITERATIONS=${2:-100}    # Default to 100 iterations if not specified

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Records: $RECORDS"
echo "  Iterations: $ITERATIONS"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Python dependencies are installed
if ! python3 -c "import qdrant_client" > /dev/null 2>&1; then
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
fi

# Start Qdrant
echo "Starting Qdrant database..."
docker-compose up -d

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
sleep 30

# Check if Qdrant is responding
echo "Checking Qdrant health..."
for i in {1..30}; do
    if curl -s http://$HOST:$PORT/health > /dev/null 2>&1; then
        echo "Qdrant is ready!"
        break
    fi
    echo "Waiting for Qdrant... ($i/30)"
    sleep 2
done

if ! curl -s http://$HOST:$PORT/health > /dev/null 2>&1; then
    echo "Error: Qdrant is not responding. Check Docker logs:"
    docker-compose logs qdrant
    exit 1
fi

# Create results directory
mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/run_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Starting test pipeline..."
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# 1. Populate Database
echo "=========================================="
echo "Step 1: Populating Database"
echo "=========================================="
python3 populate_database.py \
    --host $HOST \
    --port $PORT \
    --records $RECORDS \
    --workers 4 \
    --vector-dim 768

echo "Database population completed!"
echo ""

# 2. Run Read Benchmark
echo "=========================================="
echo "Step 2: Read Performance Benchmark"
echo "=========================================="
python3 benchmark_reads.py \
    --host $HOST \
    --port $PORT \
    --collection test_vectors \
    --iterations $ITERATIONS \
    --output "$RESULTS_DIR/read_benchmark_results.json"

echo "Read benchmark completed!"
echo ""

# 3. Run Write Benchmark
echo "=========================================="
echo "Step 3: Write Performance Benchmark"
echo "=========================================="
python3 benchmark_writes.py \
    --host $HOST \
    --port $PORT \
    --collection write_test_vectors \
    --iterations $ITERATIONS \
    --output "$RESULTS_DIR/write_benchmark_results.json" \
    --cleanup

echo "Write benchmark completed!"
echo ""

# 4. Run Comprehensive Benchmark
echo "=========================================="
echo "Step 4: Comprehensive Benchmark"
echo "=========================================="
python3 benchmark_comprehensive.py \
    --host $HOST \
    --port $PORT \
    --read-collection test_vectors \
    --write-collection write_test_vectors \
    --iterations $ITERATIONS \
    --load-duration 300 \
    --output "$RESULTS_DIR/comprehensive_benchmark_results.json"

echo "Comprehensive benchmark completed!"
echo ""

# 5. Generate Summary Report
echo "=========================================="
echo "Step 5: Generating Summary Report"
echo "=========================================="

cat > "$RESULTS_DIR/summary_report.md" << EOF
# Vector Database Test Results

**Test Run:** $TIMESTAMP
**Records:** $RECORDS
**Iterations:** $ITERATIONS
**Host:** $HOST:$PORT

## Files Generated

- \`read_benchmark_results.json\` - Read performance metrics
- \`write_benchmark_results.json\` - Write performance metrics  
- \`comprehensive_benchmark_results.json\` - Complete benchmark results
- \`summary_report.md\` - This summary report

## Quick Stats

EOF

# Extract key metrics from results
if [ -f "$RESULTS_DIR/read_benchmark_results.json" ]; then
    echo "### Read Performance" >> "$RESULTS_DIR/summary_report.md"
    echo "" >> "$RESULTS_DIR/summary_report.md"
    
    # Extract single search performance
    SINGLE_SEARCH_MEAN=$(python3 -c "
import json
with open('$RESULTS_DIR/read_benchmark_results.json') as f:
    data = json.load(f)
    if 'single_search' in data:
        print(f\"{data['single_search']['mean']:.4f}\")
    else:
        print('N/A')
")
    echo "- Single Search Mean: ${SINGLE_SEARCH_MEAN}s" >> "$RESULTS_DIR/summary_report.md"
    
    # Extract concurrent search QPS
    CONCURRENT_QPS=$(python3 -c "
import json
with open('$RESULTS_DIR/read_benchmark_results.json') as f:
    data = json.load(f)
    if 'concurrent_search' in data:
        print(f\"{data['concurrent_search']['qps']:.2f}\")
    else:
        print('N/A')
")
    echo "- Concurrent Search QPS: ${CONCURRENT_QPS}" >> "$RESULTS_DIR/summary_report.md"
    echo "" >> "$RESULTS_DIR/summary_report.md"
fi

if [ -f "$RESULTS_DIR/write_benchmark_results.json" ]; then
    echo "### Write Performance" >> "$RESULTS_DIR/summary_report.md"
    echo "" >> "$RESULTS_DIR/summary_report.md"
    
    # Extract batch insert performance
    BATCH_INSERT_THROUGHPUT=$(python3 -c "
import json
with open('$RESULTS_DIR/write_benchmark_results.json') as f:
    data = json.load(f)
    if 'batch_insert_1000' in data:
        print(f\"{data['batch_insert_1000']['throughput']:.2f}\")
    else:
        print('N/A')
")
    echo "- Batch Insert (1000) Throughput: ${BATCH_INSERT_THROUGHPUT} points/sec" >> "$RESULTS_DIR/summary_report.md"
    echo "" >> "$RESULTS_DIR/summary_report.md"
fi

echo "## Next Steps" >> "$RESULTS_DIR/summary_report.md"
echo "" >> "$RESULTS_DIR/summary_report.md"
echo "1. Review the JSON result files for detailed metrics" >> "$RESULTS_DIR/summary_report.md"
echo "2. Compare results with previous runs" >> "$RESULTS_DIR/summary_report.md"
echo "3. Adjust configuration based on performance requirements" >> "$RESULTS_DIR/summary_report.md"
echo "4. Consider system resource optimization if needed" >> "$RESULTS_DIR/summary_report.md"

echo "Summary report generated: $RESULTS_DIR/summary_report.md"
echo ""

# 6. Cleanup (optional)
read -p "Do you want to stop Qdrant and clean up? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Stopping Qdrant..."
    docker-compose down
    echo "Cleanup completed!"
else
    echo "Qdrant is still running. Use 'docker-compose down' to stop it."
fi

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/summary_report.md"
echo "  python3 -m json.tool $RESULTS_DIR/read_benchmark_results.json"
echo "  python3 -m json.tool $RESULTS_DIR/write_benchmark_results.json"
echo ""

python populate_postgres.py --records $1
python populate_milvus.py --records $1
python populate_weaviate.py --records $1
python populate_qdrant.py --records $1
#!/usr/bin/env python3
"""
Simple Qdrant Web UI
A basic web interface to explore your Qdrant vector database.
"""

from flask import Flask, render_template, request, jsonify
import requests
import json
import numpy as np
from qdrant_client import QdrantClient
import argparse

app = Flask(__name__)

# Global client
client = None

def init_client(host="localhost", port=6333):
    global client
    client = QdrantClient(host=host, port=port)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/collections')
def get_collections():
    try:
        collections = client.get_collections()
        return jsonify({
            "status": "success",
            "collections": [{"name": col.name} for col in collections.collections]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/collection/<collection_name>')
def get_collection_info(collection_name):
    try:
        info = client.get_collection(collection_name)
        return jsonify({
            "status": "success",
            "info": {
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "indexed_vectors": info.indexed_vectors_count
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/collection/<collection_name>/points')
def get_points(collection_name):
    try:
        limit = request.args.get('limit', 10, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get points using scroll
        result = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False  # Don't return vectors for performance
        )
        
        points = []
        for point in result[0]:
            points.append({
                "id": point.id,
                "payload": point.payload
            })
        
        return jsonify({
            "status": "success",
            "points": points,
            "next_page_offset": result[1] if result[1] else None
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/collection/<collection_name>/search', methods=['POST'])
def search_points(collection_name):
    try:
        data = request.json
        query_vector = data.get('vector')
        limit = data.get('limit', 10)
        
        if not query_vector:
            return jsonify({"status": "error", "message": "Query vector is required"})
        
        # Perform search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        search_results = []
        for result in results:
            search_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        return jsonify({
            "status": "success",
            "results": search_results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/collection/<collection_name>/random_vector')
def get_random_vector(collection_name):
    try:
        # Get collection info to know vector size
        info = client.get_collection(collection_name)
        vector_size = info.config.params.vectors.size
        
        # Generate random normalized vector
        vector = np.random.normal(0, 1, vector_size)
        vector = vector / np.linalg.norm(vector)
        
        return jsonify({
            "status": "success",
            "vector": vector.tolist()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Qdrant Web UI")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--ui-port", type=int, default=5000, help="Web UI port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize Qdrant client
    init_client(args.host, args.port)
    
    print(f"Starting Qdrant Web UI on http://localhost:{args.ui_port}")
    print(f"Connected to Qdrant at {args.host}:{args.port}")
    
    app.run(host='0.0.0.0', port=args.ui_port, debug=args.debug)

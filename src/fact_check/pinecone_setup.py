#!/usr/bin/env python3
"""
Script to set up Pinecone index for text and image embeddings
"""

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing required API key. Please set PINECONE_API_KEY in .env file")

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "med-cite-index"

# Create Pinecone index if it doesn't exist
def create_index():
    print(f"Setting up Pinecone index: {INDEX_NAME}")
    
    if INDEX_NAME in pc.list_indexes().names():
        print(f"Index {INDEX_NAME} already exists")
        
        # Check if the existing index supports sparse vectors
        index_info = pc.describe_index(INDEX_NAME)
        print(f"Index info: {index_info}")
        
        # Log index details to debug sparse vector support
        if hasattr(index_info, 'metric') and index_info.metric != 'dotproduct':
            print(f"Warning: Existing index uses {index_info.metric} metric which may not support sparse vectors.")
            print("Consider deleting the index and recreating with dotproduct metric.")
    else:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # Gemini embedding dimension
            metric="dotproduct",  # Use dotproduct to support sparse vectors
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Successfully created index: {INDEX_NAME}")
    
    # Return the index
    return pc.Index(INDEX_NAME)

if __name__ == "__main__":
    # Create the index
    index = create_index()
    print("Pinecone setup complete!")
    
    # Print index stats
    try:
        stats = index.describe_index_stats()
        print(f"Index statistics: {stats}")
        print(f"Total vectors: {stats.total_vector_count}")
        print(f"Namespaces: {stats.namespaces}")
    except Exception as e:
        print(f"Error getting index stats: {e}") 
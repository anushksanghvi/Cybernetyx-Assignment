from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from chromadb.config import Settings
from chromadb import Client
from sentence_transformers import SentenceTransformer
from doc_parser import parse_document
from typing import List
import asyncio

app = FastAPI()

# Initialize ChromaDB client with persistent storage
chroma_client = Client(Settings(persist_directory="./chroma_db"))
# Create a collection for storing document embeddings
collection = chroma_client.create_collection("document_store")

# Load sentence-transformer model (using CPU-friendly all-MiniLM-L6-v2)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Ingest document function (asynchronous)
async def ingest_document(file: UploadFile):
    """Ingest a document, generate its embedding, and store it in ChromaDB."""
    try:
        # Parse the document to extract text
        text = await parse_document(file)
        if not text:
            raise ValueError("Document is empty.")
        
        # Generate the embedding for the document text
        embedding = model.encode(text)
        
        # Add the document text and its embedding to ChromaDB
        collection.add(
            documents=[text], 
            embeddings=[embedding], 
            metadatas=[{"filename": file.filename}]
        )
        
        return {"message": "Document ingested successfully", "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

# Endpoint to upload and ingest documents
@app.post("/ingest/")
async def upload_file(file: UploadFile = File(...)):
    """API endpoint to ingest a document."""
    return await ingest_document(file)

# Query documents function (asynchronous)
async def query_documents(query: str):
    """Query documents in ChromaDB by embedding the query and finding similar documents."""
    try:
        # Generate embedding for the search query
        query_embedding = model.encode(query)
        
        # Query the ChromaDB collection
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        
        # Extract relevant document texts and metadata
        documents = [{"document": doc, "metadata": meta} for doc, meta in zip(results["documents"], results["metadatas"])]
        
        return {"query": query, "results": documents}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

# Endpoint to query documents based on a text query
@app.post("/query/")
async def search_documents(query: str):
    """API endpoint to search for relevant documents based on a query."""
    return await query_documents(query)

# Import necessary libraries
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from typing import List

# Constants
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Change as needed

# Step 1: Load the embedding model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Function to compute embeddings
def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Step 2: Initialize and populate FAISS index
index = faiss.IndexFlatL2(384)  # Assuming embedding size of 384 (MiniLM model)

# Example corpus
corpus = [
    "What is the return policy?",
    "How can I reset my password?",
    "Where can I find my order history?",
    "What are the shipping options available?",
]

# Compute embeddings and add to FAISS index
corpus_embeddings = compute_embeddings(corpus)
index.add(corpus_embeddings)

# Step 3: Implement retrieval function
def retrieve(query: str, k: int = 3) -> List[str]:
    """Retrieve top-k relevant documents for a query."""
    query_embedding = compute_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    return [corpus[i] for i in indices[0]]

# Step 4: Use a generative model for answer synthesis
generator = pipeline("text2text-generation", model="t5-small")  # Adjust model as needed

def generate_response(query: str) -> str:
    """Generate a response using retrieved documents and generative AI."""
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\nQuery: {query}\nAnswer:"
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

# Test the pipeline
query = "How do I check my past orders?"
response = generate_response(query)
print("Generated Response:", response)

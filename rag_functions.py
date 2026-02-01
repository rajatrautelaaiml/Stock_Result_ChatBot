#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.vectorstores import FAISS
import torch
import requests
import fitz  # PyMuPDF
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
import pandas as pd
import re
from langchain_groq import ChatGroq


# In[ ]:


# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[ ]:


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


# In[ ]:


def clean_text(text):
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (optional)
    text = re.sub(r'[^a-zA-Z0-9\s.,₹%]', '', text)
    
    return text


# In[ ]:


# Step 5: Split text into overlapping chunks
def preprocess_text(text, chunk_size=500, overlap=100, remove_stopwords=True, lemmatize=True):
    """
    Clean and split text into overlapping chunks.
    
    Args:
        text (str): The input text.
        chunk_size (int): Size of each chunk in characters.
        overlap (int): Number of overlapping characters between chunks.
        remove_stopwords (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to lemmatize words.
    
    Returns:
        list: List of cleaned and overlapping text chunks.
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Split into overlapping chunks
    chunks = []
    start = 0
    while start < len(cleaned_text):
        end = start + chunk_size
        chunk = cleaned_text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # Move start by chunk_size minus overlap
    
    return chunks


# In[ ]:


# Wrap SentenceTransformer in LangChain's HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# In[ ]:


def create_vector_store(text_chunks):
    """
    Create a vector store from cleaned text chunks.
    
    Args:
        text_chunks (list): List of cleaned text chunks.
    
    Returns:
        FAISS: Vector store for semantic search.
    """
    # Create FAISS vector store
    vector_store = FAISS.from_texts(
        texts=text_chunks,  # List of text chunks
        embedding=embedding_model  # LangChain Embeddings object
    )
    return vector_store


# In[ ]:


def retrieve_relevant_chunks(query, vector_store, k=3):
    """
    Retrieve the top-k relevant chunks for a query.
    
    Args:
        query (str): The user's query.
        vector_store (FAISS): The vector store.
        k (int): Number of chunks to retrieve.
    
    Returns:
        list: List of relevant chunks (Document objects).
    """
    relevant_chunks = vector_store.similarity_search(query, k=k)
    return relevant_chunks


# In[ ]:


# Set your Groq API key and Chat API URL
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Replace with the correct Chat API URL

def generate_answer_with_groq(query, context):
    """
    Generate an answer using Groq's Chat API.
    
    Args:
        query (str): The user's query.
        context (str): Retrieved context.
    
    Returns:
        str: Generated answer.
    """
    # Create the prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
    ]
    
    # Prepare the API request payload
    payload = {
        "model": "llama-3.3-70b-versatile",  # Replace with the correct model name (e.g., "mistral", "mixtral")
        "messages": messages,
        "max_tokens": 2000,  # Adjust as needed
        "temperature": 0.1,  # Adjust as needed
        "top_p": 1  # Adjust as needed
    }
    
    # Set headers with API key
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Send the request to Groq Chat API
    print("Sending request to Groq Chat API...")  # Debug statement
    print("Payload:", payload)  # Debug statement
    response = requests.post(GROQ_CHAT_API_URL, json=payload, headers=headers)
    
    # Check for errors
    if response.status_code != 200:
        print("Groq Chat API request failed!")  # Debug statement
        print("Status Code:", response.status_code)  # Debug statement
        print("Response Text:", response.text)  # Debug statement
        raise Exception(f"Groq Chat API request failed: {response.status_code}, {response.text}")
    
    # Extract the generated answer
    answer = response.json()["choices"][0]["message"]["content"]
    return answer


# In[ ]:


def rag_chatbot(query, vector_store):
    """
    Generate an answer using the RAG pipeline with Groq Chat API.
    
    Args:
        query (str): The user's query.
        vector_store (FAISS): The vector store.
    
    Returns:
        str: Generated answer.
    """
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, vector_store)
    
    # Construct context by joining the page_content of each chunk
    context = " ".join([chunk.page_content for chunk in relevant_chunks])
    
    # Generate answer using Groq Chat API
    answer = generate_answer_with_groq(query, context)
    return answer


# In[ ]:





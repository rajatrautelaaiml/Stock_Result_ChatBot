#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from rag_functions import extract_text_from_pdf, clean_text, preprocess_text, create_vector_store, retrieve_relevant_chunks, generate_answer_with_groq, rag_chatbot

# Title of the app
st.title("RAG Chatbot for PDFs")

# Sidebar for additional options
with st.sidebar:
    st.header("About")
    st.write("This is a RAG chatbot that answers questions based on the uploaded PDF.")
    st.write("Upload a PDF and start chatting!")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Preprocess text and create vector store
    text_chunks = preprocess_text(pdf_text)
    vector_store = create_vector_store(text_chunks)
    
    # Display success message
    st.success("PDF uploaded and processed successfully!")

    # Display extracted text (optional)
    if st.checkbox("Show extracted text"):
        st.write("**Extracted Text:**")
        st.write(pdf_text)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat interface
    st.header("Chat with the PDF")
    user_query = st.chat_input("Ask a question about the PDF:")
    
    if user_query:
        # Add user query to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Get the answer from the RAG chatbot
        answer = rag_chatbot(user_query, vector_store)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Display the answer
        with st.chat_message("assistant"):
            st.write(answer)


# In[ ]:





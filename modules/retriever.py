import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def create_retriever(splits):
    """
    Create a retriever using ChromaDB and BioBERT embeddings, caching the result.
    
    Args:
        splits (list): List of document chunks from the dataset.
    
    Returns:
        Retriever: LangChain retriever object for similarity search.
    """
    # Use BioBERT embeddings for clinical text
    embeddings = HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    
    # Create Chroma vector store
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Top 5 matches
    return retriever
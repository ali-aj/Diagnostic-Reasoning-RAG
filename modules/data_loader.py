import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource
def load_dataset(dataset_path):
    """
    Load and preprocess the dataset from a directory, caching the result.
    
    Args:
        dataset_path (str): Path to the directory containing JSON files.
    
    Returns:
        list: List of document chunks ready for indexing.
    """
    # Load all JSON files from the directory
    loader = DirectoryLoader(dataset_path, glob="**/*.json")
    docs = loader.load()
    
    # Split documents into chunks for efficient indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits
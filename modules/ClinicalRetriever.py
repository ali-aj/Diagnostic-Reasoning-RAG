from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import torch

class ClinicalRetriever:
    def __init__(self):
        # Use a more reliable embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len
        )
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create vector store with error handling"""
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        try:
            # Verify documents have content
            valid_docs = [doc for doc in documents if doc.page_content.strip()]
            if not valid_docs:
                raise ValueError("All documents are empty")
            
            # Split documents
            chunks = self.text_splitter.split_documents(valid_docs)
            if not chunks:
                raise ValueError("No chunks generated after splitting")
            
            # Verify embeddings work
            test_text = chunks[0].page_content[:100]  # First 100 chars of first chunk
            test_embedding = self.embedding_model.embed_query(test_text)
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding model failed to create test embedding")
            
            # Create vector store
            return FAISS.from_documents(chunks, self.embedding_model)
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            print(f"First document content: {documents[0].page_content[:200] if documents else 'No documents'}")
            raise
    
    def create_bm25_retriever(self, documents: List[Document]) -> BM25Retriever:
        """Create BM25 retriever"""
        texts = [doc.page_content for doc in documents]
        return BM25Retriever.from_texts(texts)
    
    def create_ensemble_retriever(self, vector_store: FAISS, bm25_retriever: BM25Retriever) -> EnsembleRetriever:
        """Create hybrid retriever combining semantic and keyword search"""
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.4, 0.6]
        )
        return ensemble_retriever
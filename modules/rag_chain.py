from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

def create_rag_chain(retriever):
    """
    Create a RAG chain combining retrieval and generation.
    
    Args:
        retriever: LangChain retriever object.
    
    Returns:
        RetrievalQA: Chain object for query answering.
    """
    # Use BART-large for generation
    llm = HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="facebook/bart-large"))
    
    # Create RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return rag_chain
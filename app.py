import streamlit as st
from modules.data_loader import load_dataset
from modules.retriever import create_retriever
from modules.rag_chain import create_rag_chain

# Initialize RAG system once, using caching
if "rag_chain" not in st.session_state:
    dataset_path = "diagnostic_kg/Diagnosis_flowchart/"  
    splits = load_dataset(dataset_path)
    retriever = create_retriever(splits)
    rag_chain = create_rag_chain(retriever)
    st.session_state.rag_chain = rag_chain

# Streamlit UI
st.title("Clinical RAG System")
st.write("Ask a question based on the dataset.")

query = st.text_input("Enter your query:", placeholder="e.g., What are common symptoms?")
if st.button("Search"):
    with st.spinner("Generating answer..."):
        answer = st.session_state.rag_chain.run(query)
    st.write("### Answer")
    st.write(answer)
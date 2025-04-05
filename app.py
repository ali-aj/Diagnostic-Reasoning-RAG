import streamlit as st
from modules.MIMICDataLoader import MIMICDataLoader
from modules.ClinicalRetriever import ClinicalRetriever
from modules.ClinicalQA import ClinicalQA
from pathlib import Path
import torch

def load_model_safely():
    """Helper function to manage system resources"""
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def verify_data_paths(kg_dir: str, samples_dir: str) -> bool:
    """Verify the directory structure matches expectations"""
    kg_path = Path(kg_dir) / "Diagnosis_flowchart"
    samples_path = Path(samples_dir) / "Finished"
    
    if not kg_path.exists():
        print(f"Knowledge graph directory not found: {kg_path}")
        return False
    
    if not samples_path.exists():
        print(f"Samples directory not found: {samples_path}")
        return False
    
    # Check for JSON files in kg directory
    kg_files = list(kg_path.glob("*.json"))
    if not kg_files:
        print(f"No JSON files found in {kg_path}")
        return False
    
    # Check for at least one disease category in samples
    sample_dirs = [d for d in samples_path.iterdir() if d.is_dir()]
    if not sample_dirs:
        print(f"No disease categories found in {samples_path}")
        return False
    
    # Check for PDD subdirectories
    pdd_dirs = []
    for disease_dir in sample_dirs:
        pdd_dirs.extend([d for d in disease_dir.iterdir() if d.is_dir()])
    
    if not pdd_dirs:
        print(f"No PDD categories found in sample directories")
        return False
    
    # Check for actual note files
    note_files = []
    for pdd_dir in pdd_dirs:
        note_files.extend(list(pdd_dir.glob("*.json")))
    
    if not note_files:
        print(f"No JSON note files found in PDD directories")
        return False
    
    print(f"Data paths verified:")
    print(f"- Found {len(kg_files)} knowledge graphs")
    print(f"- Found {len(sample_dirs)} disease categories")
    print(f"- Found {len(pdd_dirs)} PDD categories")
    print(f"- Found {len(note_files)} annotated notes")
    
    return True

# Initialize RAG system once, using caching
if "rag_chain" not in st.session_state and verify_data_paths("diagnostic_kg", "samples"):
    load_model_safely()
    
    data_loader = MIMICDataLoader("diagnostic_kg", "samples")
    retriever_setup = ClinicalRetriever()
    qa_system = ClinicalQA()
    
    # Load data
    print("Loading knowledge graphs...")
    kg_docs = data_loader.load_knowledge_graphs()
    # print(f"Knowledge Graphs Start...")
    
    # for doc in kg_docs:
    #     print(f"- {doc.metadata['source']} ({doc.metadata['disease_category']}, {doc.metadata['type']})")

    # print(f"Knowledge Graphs End...")
    print("Loading annotated notes...")
    note_docs = data_loader.load_annotated_notes()
    all_docs = kg_docs + note_docs
    
    # Create retrieval systems
    print("Creating vector store...")
    vector_store = retriever_setup.create_vector_store(all_docs)
    print("Creating BM25 retriever...")
    bm25_retriever = retriever_setup.create_bm25_retriever(all_docs)
    print("Creating ensemble retriever...")
    ensemble_retriever = retriever_setup.create_ensemble_retriever(vector_store, bm25_retriever)
    
    # Create QA chain
    print("Initializing QA system...")
    rag_chain = qa_system.create_qa_chain(ensemble_retriever)

    st.session_state.rag_chain = rag_chain

# Streamlit UI
st.title("Clinical RAG System")
st.write("Ask a question based on the dataset.")

query = st.text_input("Enter your query:", placeholder="e.g., What are common symptoms?", key="query")
if st.button("Search"):
    with st.spinner("Generating answer..."):
        result = st.session_state.rag_chain.invoke({"query": query})
        for doc in result['source_documents']:
            if 'source' in doc.metadata:
                print(f"- {doc.metadata['source']} ({doc.metadata.get('type', 'unknown')})")
    st.write("### Answer")
    st.write(result['result'])
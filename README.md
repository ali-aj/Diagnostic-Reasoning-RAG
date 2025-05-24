# Diagnostic-Reasoning-RAG

A Clinical Decision Support System using Retrieval Augmented Generation (RAG) with medical knowledge graphs and annotated clinical notes.

## Overview

This project implements a RAG-based system that combines structured medical knowledge with clinical case notes to provide evidence-based clinical decision support. It uses the Gemma language model via Ollama for generation and a hybrid retrieval system combining semantic search (FAISS) with keyword search (BM25).

## Features

- Knowledge Graph Integration
- Clinical Notes Processing
- Hybrid Retrieval System
- Interactive Web Interface
- Evidence-based Responses

## Prerequisites

- Python 3.8+
- Ollama with Gemma model installed
- CUDA-capable GPU (optional, improves performance)

## Installation

1. Clone the repository:
```sh
git clone https://github.com/ali-aj/Diagnostic-Reasoning-RAG.git
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Ensure Ollama is installed with the Gemma model:
```sh
ollama pull gemma3:4b
```

## Project Structure

```
.
├── app.py                  # Streamlit web application
├── diagnostic_kg/         # Medical knowledge graphs
│   └── Diagnosis_flowchart/
├── modules/
│   ├── ClinicalQA.py      # QA chain implementation
│   ├── ClinicalRetriever.py  # Retrieval system
│   └── MIMICDataLoader.py   # Data loading utilities
└── samples/              # Clinical case notes
    └── Finished/
```

## Usage

1. Start the web interface:
```sh
streamlit run app.py
```

2. Enter clinical queries in the text input field
3. Review the structured responses including:
   - Clinical Summary
   - Diagnostic Considerations
   - Supporting Evidence
   - Differential Diagnosis
   - Recommended Next Steps

## Technical Details

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Language Model**: Gemma 4B via Ollama
- **Retrieval**: Ensemble of FAISS (60%) and BM25 (40%)

## Data Structure

The system uses two main data sources:
1. Diagnostic Knowledge Graphs (.json)
   - Disease categories
   - Diagnostic pathways
   - Clinical knowledge

2. Annotated Clinical Notes
   - Patient history
   - Clinical observations
   - Diagnostic chains
   - Clinical deductions
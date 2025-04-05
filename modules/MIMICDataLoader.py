import json
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document

class MIMICDataLoader:
    def __init__(self, kg_dir: str, samples_dir: str):
        self.kg_dir = Path(kg_dir) / "Diagnosis_flowchart"
        self.samples_dir = Path(samples_dir) / "Finished"
    
    def load_knowledge_graphs(self) -> List[Document]:
        """Load knowledge graphs with proper path handling"""
        docs = []
        for kg_file in self.kg_dir.glob("*.json"):
            try:
                with open(kg_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    disease = kg_file.stem.replace("{} ", "")  # Clean filename
                    
                    diagnostic_text = self._flatten_diagnostic_tree(data["diagnostic"])
                    knowledge_text = self._process_knowledge_sections(data["knowledge"])
                    
                    content = f"Disease Category: {disease}\n\nDiagnostic Path:\n{diagnostic_text}\n\nClinical Knowledge:\n{knowledge_text}"
                    
                    metadata = {
                        "source": str(kg_file),
                        "disease_category": disease,
                        "type": "knowledge_graph"
                    }
                    docs.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f"Error loading {kg_file}: {str(e)}")
                continue
        return docs
    
    def _flatten_diagnostic_tree(self, tree: Dict, level: int = 0) -> str:
        """Recursively flatten diagnostic tree structure"""
        text = ""
        for node, children in tree.items():
            text += "  " * level + f"- {node}\n"
            if children:
                text += self._flatten_diagnostic_tree(children, level + 1)
        return text
    
    def _process_knowledge_sections(self, knowledge: Dict) -> str:
        """Convert knowledge sections to readable text"""
        sections = []
        for category, details in knowledge.items():
            if isinstance(details, dict):
                for subcat, content in details.items():
                    sections.append(f"{category} - {subcat}: {content}")
            else:
                sections.append(f"{category}: {details}")
        return "\n".join(sections)
    
    def load_annotated_notes(self) -> List[Document]:
        """Load annotated notes with corrected path structure"""
        docs = []
        for disease_dir in self.samples_dir.iterdir():
            if disease_dir.is_dir():
                for pdd_dir in disease_dir.iterdir():
                    if pdd_dir.is_dir():
                        for note_file in pdd_dir.glob("*.json"):
                            try:
                                with open(note_file, 'r', encoding='utf-8') as f:
                                    note_data = json.load(f)
                                    content = self._process_note(note_data)
                                    
                                    metadata = {
                                        "source": str(note_file),
                                        "disease_category": disease_dir.name,
                                        "pdd_category": pdd_dir.name,
                                        "type": "annotated_note"
                                    }
                                    docs.append(Document(page_content=content, metadata=metadata))
                            except Exception as e:
                                print(f"Error loading {note_file}: {str(e)}")
                                continue
        return docs
    
    def _process_note(self, note_data: Dict) -> str:
        """Extract relevant information from annotated note"""
        parts = []
        
        # Add input content (chief complaint, history, etc.)
        if "input_content" in note_data:
            for key, value in note_data["input_content"].items():
                parts.append(f"{key}: {value}")
        
        # Add diagnostic chain if available
        if "chain" in note_data:
            parts.append("\nDiagnostic Chain:")
            parts.append(" -> ".join(note_data["chain"]))
        
        # Add deductions if available
        if "GT" in note_data:
            parts.append("\nClinical Deductions:")
            for obs, deductions in note_data["GT"].items():
                for d in deductions:
                    parts.append(f"- Observation: {obs}")
                    parts.append(f"  Diagnosis: {d['d']}")
                    parts.append(f"  Rationale: {d['z']}")
                    parts.append(f"  Source: {d['r']}")
        
        return "\n".join(parts)
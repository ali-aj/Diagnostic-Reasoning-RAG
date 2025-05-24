from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever

class ClinicalQA:
    def __init__(self):
        # Initialize Gemma via Ollama
        self.llm = Ollama(
            model="gemma3:4b",
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.1,
            num_ctx=4096 
        )
        
        # Clinical prompt template optimized for Gemma
        self.prompt_template = """<start_of_turn>system
You are a clinical decision support assistant. Provide accurate, evidence-based responses to medical queries.
Structure your answers clearly and cite sources when possible.<end_of_turn>
<start_of_turn>user
Clinical Context:
{context}

Question: {question}

Please respond with:
1. Clinical Summary
2. Diagnostic Considerations
3. Supporting Evidence
4. Differential Diagnosis (if applicable)
5. Recommended Next Steps<end_of_turn>
<start_of_turn>assistant
"""
    
    def create_qa_chain(self, retriever: EnsembleRetriever) -> RetrievalQA:
        """Create QA chain with Gemma-specific prompt"""
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
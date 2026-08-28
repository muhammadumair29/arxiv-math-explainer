import os
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGPipeline:
    """
    Manages the Vector Database, Embeddings, and the LLM Reasoning chain.
    Configured specifically for handling dense mathematical context and LaTeX outputs.
    """
    
    def __init__(self, api_key: str = None):
        # Allow passing API key directly or reading from environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")
            
        # Initialize Embeddings and LLM
        # Using Gemini 1.5 Flash or Pro depending on your deployment preference
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=self.api_key,
            temperature=0.2 # Low temperature for factual mathematical accuracy
        )
        
        self.vector_store = None
        self.retriever = None

    def build_vector_store(self, chunks: List[Document]):
        """
        Takes math-aware document chunks and indexes them into an in-memory ChromaDB.
        """
        try:
            # Ephemeral ChromaDB for the session (refreshes on new document upload)
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            # Retrieve top 4 most relevant chunks
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            raise RuntimeError(f"Failed to build vector store: {str(e)}")

    def get_qa_chain(self):
        """
        Constructs the RAG chain with a strict mathematical reasoning prompt.
        """
        if not self.retriever:
            raise ValueError("Vector store not initialized. Please upload and process a document first.")

        # The System Prompt designed to enforce academic rigor and LaTeX formatting
        prompt_template = """
        You are an applied mathematics professor. Your task is to explain complex mathematical 
        concepts, equations, and proofs to an advanced undergraduate student. 
        
        Use the following retrieved context from an academic paper to answer the question. 
        If the context does not contain the answer, state clearly that you do not know based on the provided text.
        
        CRITICAL FORMATTING RULES:
        1. Break down complex variables and equations step-by-step.
        2. You MUST return all mathematical equations and variables in standard LaTeX formatting.
        3. Enclose inline math in single dollar signs: $...$
        4. Enclose block equations in double dollar signs: $$...$$

        Context:
        {context}

        Question: {question}
        
        Professor's Explanation:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Build the LangChain LCEL (LangChain Expression Language) pipeline
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

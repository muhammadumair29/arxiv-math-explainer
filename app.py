import os
import streamlit as st
from dotenv import load_dotenv
from document_processor import MathAwareDocumentProcessor
from rag_pipeline import RAGPipeline

# Load environment variables (API Key)
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="ArXiv Math Explainer",
    page_icon="📐",
    layout="wide"
)

# --- Helper Functions ---
def sanitize_latex_for_streamlit(text: str) -> str:
    """
    Sanitizes LLM outputs to prevent Streamlit from misinterpreting LaTeX escape characters.
    This fulfills the raw string requirement by ensuring backslashes are preserved 
    when passed to st.markdown.
    """
    # Replace single backslashes with double backslashes for proper KaTeX rendering
    # Example: \frac becomes \\frac so Streamlit parses it correctly.
    sanitized = text.replace('\\', '\\\\')
    return sanitized

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# --- Sidebar UI: Document Upload & Setup ---
with st.sidebar:
    st.header("📄 Document Setup")
    uploaded_file = st.file_uploader("Upload an ArXiv PDF", type=["pdf"])
    
    if st.button("Process Document") and uploaded_file:
        with st.spinner("Extracting and processing math chunks..."):
            try:
                # 1. Initialize Processor
                doc_processor = MathAwareDocumentProcessor()
                
                # 2. Extract and Split
                pdf_bytes = uploaded_file.read()
                chunks = doc_processor.process_document(pdf_bytes, source_name=uploaded_file.name)
                
                st.success(f"Extracted {len(chunks)} math-aware chunks!")
                
                # 3. Initialize RAG Pipeline and Index
                pipeline = RAGPipeline()
                pipeline.build_vector_store(chunks)
                
                # 4. Save to session state
                st.session_state.rag_pipeline = pipeline
                st.session_state.doc_processed = True
                st.session_state.messages = [] # Clear chat history for new doc
                
                st.success("Vector database built successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    st.markdown("---")
    st.info("Ensure your `.env` file contains your `GEMINI_API_KEY`.")

# --- Main UI: Chat Interface ---
st.title("📐 ArXiv Math Explainer")
st.markdown("Upload a dense mathematical PDF, and ask the AI professor to explain the equations step-by-step.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Use sanitize_latex_for_streamlit to safely render math
        st.markdown(sanitize_latex_for_streamlit(message["content"]))

# Chat Input
if prompt := st.chat_input("Ask a question about the equations in the paper..."):
    if not st.session_state.doc_processed:
        st.warning("Please upload and process a PDF document first.")
    else:
        # 1. Add user message to state and UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing mathematical context..."):
                try:
                    # Get the RAG chain and invoke it
                    qa_chain = st.session_state.rag_pipeline.get_qa_chain()
                    response = qa_chain.invoke(prompt)
                    
                    # Apply our LaTeX sanitization function before rendering
                    safe_response = sanitize_latex_for_streamlit(response)
                    st.markdown(safe_response)
                    
                    # Save raw response to state
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

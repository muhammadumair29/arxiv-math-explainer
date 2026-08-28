import re
import fitz  # PyMuPDF
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class MathAwareDocumentProcessor:
    """
    Handles PDF extraction and math-aware text chunking.
    Prioritizes preserving LaTeX equation blocks during the splitting process.
    """
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # This text splitter uses LaTeX and Markdown markers as separators
        # to avoid breaking mathematical contexts.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n$$",  # Block equations (start/end)
                "$$\n\n",
                "\n\n",    # Paragraph breaks
                "\n",       # Line breaks
                ". ",        # Sentences
                " ",         # Words
                ""           # Fallback
            ],
            keep_separator=True
        )

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extracts raw text from a PDF file byte stream using PyMuPDF.
        Note: For heavily formatted ArXiv papers, this relies on the PDF having 
        an embedded text layer. (For scanned docs, Nougat/Marker is required).
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_content = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                # Clean up multiple newlines that might disrupt equation blocks
                text = re.sub(r'\n{3,}', '\n\n', text)
                text_content.append(text)
                
            return "\n\n---PAGE BREAK---\n\n".join(text_content)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    def clean_and_normalize_math(self, text: str) -> str:
        """
        Attempts basic normalization of math environments if the PDF text layer 
        was slightly garbled. (A highly complex implementation would use an OCR model here).
        """
        # Ensure block equations have proper spacing for the text splitter
        text = re.sub(r'(?<!\n)\$\$', '\n\n$$', text)
        text = re.sub(r'\$\$(?!\n)', '$$\n\n', text)
        return text

    def process_document(self, pdf_bytes: bytes, source_name: str = "uploaded_document") -> List[Document]:
        """
        End-to-end processing: Extract text -> Normalize -> Split into math-aware chunks.
        """
        if not pdf_bytes:
            raise ValueError("No PDF bytes provided for processing.")

        raw_text = self.extract_text_from_pdf(pdf_bytes)
        normalized_text = self.clean_and_normalize_math(raw_text)
        
        # Create a single LangChain document first
        full_doc = Document(
            page_content=normalized_text, 
            metadata={"source": source_name}
        )
        
        # Split into smaller chunks while preserving equation blocks
        chunks = self.splitter.split_documents([full_doc])
        return chunks

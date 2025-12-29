"""
PDF Processing Module
Handles PDF text extraction and chunking for RAG pipeline
"""
import pdfplumber
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    """Process PDF documents: extract text and create chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks



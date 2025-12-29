"""
RAG Chain Module
Handles retrieval-augmented generation with multiple LLM providers
"""
from typing import List, Tuple, Optional
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
import os


class RAGChain:
    """RAG chain with support for multiple LLM providers"""
    
    def __init__(self, provider: str = "gemini", temperature: float = 0.7):
        """
        Initialize RAG chain
        
        Args:
            provider: LLM provider ("gemini", "groq", "cohere")
            temperature: Model temperature
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=api_key,
                temperature=self.temperature
            )
        
        elif self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            return ChatGroq(
                model_name="llama3-8b-8192",
                groq_api_key=api_key,
                temperature=self.temperature
            )
        
        elif self.provider == "cohere":
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY not found in environment variables")
            return ChatCohere(
                cohere_api_key=api_key,
                temperature=self.temperature
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create prompt template for RAG"""
        system_template = """You are a helpful assistant that answers questions based on the provided context from PDF documents. Answer the question based on the context provided. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate."""
        
        human_template = """Context:
{context}

Question: {question}

Answer:"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    def generate_response(self, question: str, context_chunks: List[Tuple[str, float]]) -> str:
        """
        Generate response using RAG
        
        Args:
            question: User question
            context_chunks: List of (chunk_text, score) tuples from vector search
            
        Returns:
            Generated response
        """
        if not context_chunks:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Combine context chunks
        context = "\n\n".join([chunk[0] for chunk in context_chunks])
        
        # Create prompt messages
        messages = self.prompt_template.format_messages(context=context, question=question)
        
        # Generate response
        try:
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def switch_provider(self, provider: str):
        """Switch LLM provider"""
        self.provider = provider.lower()
        self.llm = self._initialize_llm()


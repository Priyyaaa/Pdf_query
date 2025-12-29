"""
PDF Query Assistant - Streamlit Application
Main application file with UI and orchestration
"""
import streamlit as st
import os
import tempfile
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
from chat_history import ChatHistory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Query Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state 
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # LLM Provider Selection
    provider = st.selectbox(
        "Select LLM Provider",
        ["gemini", "groq", "cohere"],
        index=0
    )
    
    # Temperature
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    # Number of retrieved chunks
    top_k = st.slider("Number of Context Chunks", 1, 10, 5)
    
    # Initialize RAG chain
    if st.session_state.rag_chain is None or st.session_state.rag_chain.provider != provider:
        try:
            st.session_state.rag_chain = RAGChain(provider=provider, temperature=temperature)
            st.success(f"✅ {provider.upper()} initialized")
        except Exception as e:
            st.error(f"Error initializing {provider}: {str(e)}")
            st.info("Please set API keys in .env file")
    else:
        st.session_state.rag_chain.temperature = temperature
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history.clear_history()
        st.rerun()
    
    # Display API key status
    st.divider()
    st.subheader("API Key Status")
    google_key = "✅ Set" if os.getenv("GOOGLE_API_KEY") else "❌ Not Set"
    groq_key = "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Not Set"
    cohere_key = "✅ Set" if os.getenv("COHERE_API_KEY") else "❌ Not Set"
    
    st.write(f"Google (Gemini): {google_key}")
    st.write(f"Groq: {groq_key}")
    st.write(f"Cohere: {cohere_key}")

# Main content
st.title("📚 PDF Query Assistant")
st.markdown("Upload a PDF document and ask questions about its content using AI-powered RAG.")

# PDF Upload Section
st.header("📄 Upload PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.session_state.pdf_name != uploaded_file.name:
        # Process new PDF
        with st.spinner("Processing PDF..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Extract text
                text = st.session_state.pdf_processor.extract_text(tmp_path)
                
                if not text.strip():
                    st.error("No text could be extracted from the PDF.")
                else:
                    # Chunk text
                    chunks = st.session_state.pdf_processor.chunk_text(text)
                    
                    # Build vector store
                    vector_store = VectorStore()
                    vector_store.build_index(chunks)
                    
                    # Save to session state
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    
                    st.success(f"✅ PDF processed successfully! Found {len(chunks)} text chunks.")
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
            
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.pdf_processed = False
    
    if st.session_state.pdf_processed:
        st.info(f"📄 Current document: **{st.session_state.pdf_name}**")

# Chat Interface
st.divider()
st.header("💬 Chat with Your Document")

# Display chat history
for message in st.session_state.chat_history.get_history():
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            st.write(content)
            # Display metadata if available
            if message.get("metadata", {}).get("sources"):
                with st.expander("📎 View Sources"):
                    for i, source in enumerate(message["metadata"]["sources"], 1):
                        st.text(f"Source {i}: {source[:200]}...")

# Chat input
if st.session_state.pdf_processed and st.session_state.vector_store:
    query = st.chat_input("Ask a question about the PDF...")
    
    if query:
        # Add user message to chat
        st.session_state.chat_history.add_message("user", query)
        
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Search for relevant chunks
                    relevant_chunks = st.session_state.vector_store.search(query, k=top_k)
                    
                    # Generate response using RAG
                    if st.session_state.rag_chain:
                        response = st.session_state.rag_chain.generate_response(query, relevant_chunks)
                    else:
                        response = "RAG chain not initialized. Please check API keys."
                    
                    st.write(response)
                    
                    # Add assistant response to history
                    metadata = {
                        "sources": [chunk[0] for chunk in relevant_chunks[:3]],  # Top 3 sources
                        "provider": provider
                    }
                    st.session_state.chat_history.add_message("assistant", response, metadata)
                    
                    # Show sources
                    with st.expander("📎 View Sources"):
                        for i, (chunk, score) in enumerate(relevant_chunks[:3], 1):
                            st.text(f"Source {i} (score: {score:.4f}):")
                            st.text(chunk[:300] + "...")
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.add_message("assistant", error_msg)

else:
    st.info("👆 Please upload a PDF document to start asking questions.")

# Footer
st.divider()
st.markdown("---")
st.markdown("**PDF Query Assistant** | Built with Langchain, Streamlit, FAISS, and Sentence-Transformers")



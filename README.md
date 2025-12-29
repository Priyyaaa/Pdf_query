# PDF Query Assistant

A powerful PDF document querying system built with RAG (Retrieval-Augmented Generation) architecture and natural language processing. Query your PDF documents using natural language with support for multiple AI models.

## Features

-  **PDF Processing**: Extract and process text from PDF documents using pdfplumber
-  **Semantic Search**: FAISS vector database for efficient semantic search and retrieval
-  **Multiple LLM Support**: Choose from Gemini, Groq, or Cohere AI models
-  **Persistent Chat**: Chat history is saved and persisted across sessions
-  **RAG Architecture**: Retrieval-Augmented Generation for accurate, context-aware responses
-  **Modern UI**: Beautiful Streamlit interface for easy interaction

## Architecture

```
PDF Document → Text Extraction → Chunking → Embeddings → FAISS Index
                                                              ↓
User Query → Embedding → Vector Search → Context Retrieval → RAG Chain → Response
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up API keys**:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     ```
     GOOGLE_API_KEY=your_key_here
     GROQ_API_KEY=your_key_here
     COHERE_API_KEY=your_key_here
     ```
   - You only need to set the API key for the provider you want to use

## Getting API Keys

- **Google Gemini**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Groq**: Sign up at [Groq](https://console.groq.com/) and get your API key
- **Cohere**: Sign up at [Cohere](https://cohere.com/) and get your API key

## Usage

1. **Run the application**:
```bash
streamlit run app.py
```

2. **Upload a PDF**: Click "Choose a PDF file" and select your document

3. **Select LLM Provider**: Choose your preferred AI model from the sidebar

4. **Ask Questions**: Type your question in the chat interface and get answers based on your PDF content

## Configuration

In the sidebar, you can:
- Select LLM provider (Gemini, Groq, Cohere)
- Adjust temperature (creativity of responses)
- Set number of context chunks to retrieve
- Clear chat history

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── pdf_processor.py       # PDF text extraction and chunking
├── vector_store.py        # FAISS vector store and embeddings
├── rag_chain.py          # RAG chain with multi-LLM support
├── chat_history.py       # Persistent chat history management
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment variables
└── README.md            # This file
```

## Technologies Used

- **Streamlit**: Web interface
- **Langchain**: LLM orchestration and RAG framework
- **PDFplumber**: PDF text extraction
- **Sentence-Transformers**: Text embeddings
- **FAISS**: Vector similarity search
- **Google Gemini / Groq / Cohere**: Large Language Models

## How It Works

1. **Document Processing**: When you upload a PDF, the system extracts all text and splits it into manageable chunks
2. **Embedding Generation**: Each chunk is converted into a vector embedding using a sentence transformer model
3. **Vector Indexing**: Embeddings are stored in a FAISS index for fast similarity search
4. **Query Processing**: When you ask a question, your query is embedded and compared against the document chunks
5. **Context Retrieval**: The most relevant chunks are retrieved based on semantic similarity
6. **Response Generation**: The retrieved context is passed to the LLM along with your question to generate an accurate answer

## Notes

- The FAISS index is built in memory and not persisted by default (you can modify `vector_store.py` to add persistence)
- Chat history is saved to `chat_history.json`
- Large PDFs may take some time to process initially
- Make sure you have sufficient API credits for your chosen LLM provider

## License

This project is open source and available for educational and commercial use.

## Troubleshooting

- **API Key Errors**: Make sure your `.env` file is in the project root and contains valid API keys
- **PDF Processing Errors**: Ensure the PDF is not password-protected and contains extractable text
- **Import Errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`



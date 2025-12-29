"""
Vector Store Module
Handles embeddings and FAISS vector database operations
"""
import os
import pickle
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss


class VectorStore:
    """FAISS vector store for semantic search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "faiss_index.pkl"):
        """
        Initialize vector store
        
        Args:
            model_name: Sentence transformer model name
            index_path: Path to save/load FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for text chunks
        
        Args:
            texts: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, chunks: List[str]):
        """
        Build FAISS index from text chunks
        
        Args:
            chunks: List of text chunks to index
        """
        self.chunks = chunks
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"Index built with {len(chunks)} chunks")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar chunks
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_text, distance_score)
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        # Return results with scores
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        
        return results
    
    def save(self, filepath: str = None):
        """Save index and chunks to disk"""
        save_path = filepath or self.index_path
        data = {
            'index': self.index,
            'chunks': self.chunks,
            'model_name': self.model_name,
            'dimension': self.dimension
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved to {save_path}")
    
    def load(self, filepath: str = None):
        """Load index and chunks from disk"""
        load_path = filepath or self.index_path
        if not os.path.exists(load_path):
            print(f"No existing index found at {load_path}")
            return False
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.index = data['index']
        self.chunks = data['chunks']
        self.model_name = data['model_name']
        self.dimension = data['dimension']
        
        # Reload embedding model if needed
        if self.embedding_model is None or self.embedding_model.get_sentence_embedding_dimension() != self.dimension:
            self.embedding_model = SentenceTransformer(self.model_name)
        
        print(f"Index loaded from {load_path} with {len(self.chunks)} chunks")
        return True



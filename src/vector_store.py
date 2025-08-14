import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple
from config.config import Config

class FAISSVectorStore:
    """FAISS-based vector store for similarity search"""
    
    def __init__(self):
        """Initialize FAISS vector store"""
        self.dimension = Config.VECTOR_DIMENSION
        self.index = None
        self.texts = []  # Store original texts
        self.index_path = Config.FAISS_INDEX_PATH
        
    def create_index(self):
        """Create a new FAISS index"""
        # Using IndexFlatIP for cosine similarity (Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        print(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """
        Add embeddings and corresponding texts to the index
        
        Args:
            embeddings: numpy array of embeddings
            texts: list of corresponding text chunks
        """
        if self.index is None:
            self.create_index()
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.texts.extend(texts)
        
        print(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = Config.TOP_K_RESULTS) -> Tuple[List[str], List[float]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: query embedding vector
            k: number of top results to return
            
        Returns:
            tuple of (similar_texts, similarity_scores)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Get corresponding texts
        similar_texts = [self.texts[idx] for idx in indices[0] if idx < len(self.texts)]
        similarity_scores = scores[0].tolist()
        
        return similar_texts, similarity_scores
    
    def save_index(self, filepath: str = None):
        """Save the FAISS index and texts to disk"""
        if filepath is None:
            filepath = self.index_path
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save texts
        with open(f"{filepath}_texts.pkl", 'wb') as f:
            pickle.dump(self.texts, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str = None):
        """Load FAISS index and texts from disk"""
        if filepath is None:
            filepath = self.index_path
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load texts
            with open(f"{filepath}_texts.pkl", 'rb') as f:
                self.texts = pickle.load(f)
            
            print(f"Index loaded from {filepath}. Total embeddings: {self.index.ntotal}")
            return True
            
        except FileNotFoundError:
            print(f"Index files not found at {filepath}")
            return False
    
    def get_stats(self):
        """Get statistics about the vector store"""
        if self.index is None:
            return {"total_embeddings": 0, "dimension": self.dimension}
        
        return {
            "total_embeddings": self.index.ntotal,
            "dimension": self.dimension,
            "total_texts": len(self.texts)
        }
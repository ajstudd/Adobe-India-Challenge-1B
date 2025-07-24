"""
Embedding model module using sentence-transformers for semantic similarity.
Handles loading of pre-trained models and generating embeddings for text chunks.
"""

import os
import numpy as np
import logging
from typing import List, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """Wrapper for sentence transformer model to generate embeddings"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize embedding model.
        
        Args:
            model_path: Path to local model directory. If None, uses default path.
        """
        self.logger = logging.getLogger(__name__)
        
        # Set model path
        if model_path is None:
            model_path = Path("models/all-MiniLM-L6-v2")
        
        self.model_path = Path(model_path)
        self.model = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            if self.model_path.exists():
                self.logger.info(f"Loading model from: {self.model_path}")
                self.model = SentenceTransformer(str(self.model_path))
            else:
                # Fallback: try to load from sentence-transformers cache
                self.logger.warning(f"Model path {self.model_path} not found, trying to load from cache...")
                self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Clean and prepare texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts, 
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            self.logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to encode
            
        Returns:
            Numpy array embedding
        """
        return self.encode([text])[0]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for better embedding quality.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Truncate very long texts (transformer models have token limits)
        max_length = 500  # Conservative limit for sentence transformers
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }

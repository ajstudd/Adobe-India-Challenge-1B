"""
Document ranking module using cosine similarity for semantic matching.
Ranks document chunks based on similarity to query (persona + task).
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class DocumentRanker:
    """Ranks document chunks based on semantic similarity to query"""
    
    def __init__(self, embedding_model):
        """
        Initialize document ranker.
        
        Args:
            embedding_model: Instance of EmbeddingModel for generating embeddings
        """
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
    
    def rank_chunks(self, query: str, chunks: List[Dict[str, Any]], 
                   top_sections: int = 5, top_subsections: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Rank document chunks based on semantic similarity to query.
        
        Args:
            query: Search query (persona + task)
            chunks: List of document chunks with metadata
            top_sections: Number of top heading chunks to return
            top_subsections: Number of top paragraph chunks to return
            
        Returns:
            Tuple of (extracted_sections, subsection_analysis)
        """
        if not chunks:
            self.logger.warning("No chunks to rank")
            return [], []
        
        # Separate headings and paragraphs
        headings = [chunk for chunk in chunks if chunk.get('type') == 'heading']
        paragraphs = [chunk for chunk in chunks if chunk.get('type') == 'paragraph']
        
        self.logger.info(f"Ranking {len(headings)} headings and {len(paragraphs)} paragraphs")
        
        # Rank headings for extracted_sections
        extracted_sections = self._rank_chunk_type(query, headings, top_sections)
        
        # Rank paragraphs for subsection_analysis
        subsection_analysis = self._rank_chunk_type(query, paragraphs, top_subsections)
        
        return extracted_sections, subsection_analysis
    
    def _rank_chunk_type(self, query: str, chunks: List[Dict[str, Any]], 
                        top_k: int) -> List[Dict[str, Any]]:
        """
        Rank chunks of a specific type (heading or paragraph).
        
        Args:
            query: Search query
            chunks: List of chunks to rank
            top_k: Number of top chunks to return
            
        Returns:
            List of top-k ranked chunks with scores
        """
        if not chunks:
            return []
        
        try:
            # Extract text content
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings for query and chunks
            query_embedding = self.embedding_model.encode_single(query)
            chunk_embeddings = self.embedding_model.encode(texts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                chunk_embeddings
            )[0]
            
            # Create ranked results
            ranked_chunks = []
            for i, similarity in enumerate(similarities):
                chunk = chunks[i].copy()
                chunk['score'] = round(float(similarity), 3)
                ranked_chunks.append(chunk)
            
            # Sort by similarity score (descending)
            ranked_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top-k results
            top_chunks = ranked_chunks[:top_k]
            
            # Format output (remove type field, keep text, page, score)
            result = []
            for chunk in top_chunks:
                result.append({
                    'text': chunk['text'],
                    'page': chunk['page'],
                    'score': chunk['score']
                })
            
            self.logger.debug(f"Ranked {len(chunks)} chunks, returning top {len(result)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error ranking chunks: {e}")
            return []
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Similarity matrix as numpy array
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.embedding_model.encode(texts)
            similarity_matrix = cosine_similarity(embeddings)
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error computing similarity matrix: {e}")
            return np.array([])
    
    def find_similar_chunks(self, target_text: str, chunks: List[Dict[str, Any]], 
                           threshold: float = 0.7, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a target text.
        
        Args:
            target_text: Text to find similar chunks for
            chunks: List of chunks to search in
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of similar chunks with similarity scores
        """
        if not chunks:
            return []
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            target_embedding = self.embedding_model.encode_single(target_text)
            chunk_embeddings = self.embedding_model.encode(texts)
            
            # Calculate similarities
            similarities = cosine_similarity(
                target_embedding.reshape(1, -1), 
                chunk_embeddings
            )[0]
            
            # Filter by threshold and create results
            similar_chunks = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    chunk = chunks[i].copy()
                    chunk['similarity'] = round(float(similarity), 3)
                    similar_chunks.append(chunk)
            
            # Sort by similarity and limit results
            similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_chunks[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error finding similar chunks: {e}")
            return []

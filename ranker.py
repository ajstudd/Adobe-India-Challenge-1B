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
                   min_similarity: float = 0.3, max_sections: int = 50, max_subsections: int = 50) -> Tuple[List[Dict], List[Dict]]:
        """
        Intelligently rank and extract document chunks based on semantic similarity.
        Automatically determines the optimal amount of content to extract.
        
        Args:
            query: Search query (persona + task)
            chunks: List of document chunks with metadata
            min_similarity: Minimum similarity threshold (auto-tuned based on content quality)
            max_sections: Maximum sections (scales with document size)
            max_subsections: Maximum subsections (scales with document size)
            
        Returns:
            Tuple of (extracted_sections, subsection_analysis)
        """
        if not chunks:
            self.logger.warning("No chunks to rank")
            return [], []
        
        headings = [chunk for chunk in chunks if chunk.get('type') == 'heading']
        paragraphs = [chunk for chunk in chunks if chunk.get('type') == 'paragraph']
        
        self.logger.info(f"Ranking {len(headings)} headings and {len(paragraphs)} paragraphs")
        
        extracted_sections = self._rank_headings(query, headings, min_similarity, max_sections)
        
        subsection_analysis = self._rank_paragraphs(query, paragraphs, min_similarity, max_subsections)
        
        return extracted_sections, subsection_analysis
    
    def _rank_headings(self, query: str, headings: List[Dict[str, Any]], 
                      min_similarity: float, max_results: int) -> List[Dict[str, Any]]:
        """
        Rank heading chunks for extracted_sections output format.
        Uses similarity threshold to include all relevant headings.
        
        Args:
            query: Search query
            headings: List of heading chunks to rank
            min_similarity: Minimum similarity threshold to include content
            max_results: Maximum number of results to prevent overwhelming output
            
        Returns:
            List of ranked headings above similarity threshold in output schema format
        """
        if not headings:
            return []
        
        try:
            texts = [chunk['text'] for chunk in headings]
            
            # Generate embeddings for query and chunks
            query_embedding = self.embedding_model.encode_single(query)
            chunk_embeddings = self.embedding_model.encode(texts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                chunk_embeddings
            )[0]
            
            # Create ranked results with relevance boosting
            ranked_chunks = []
            for i, similarity in enumerate(similarities):
                chunk = headings[i].copy()
                # Apply content relevance boost
                boost = self._calculate_content_relevance_boost(chunk['text'], query)
                boosted_similarity = min(1.0, similarity + boost)
                chunk['similarity'] = float(boosted_similarity)
                chunk['original_similarity'] = float(similarity)
                chunk['relevance_boost'] = float(boost)
                ranked_chunks.append(chunk)
            
            # Sort by similarity score (descending)
            ranked_chunks.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Intelligent filtering: adjust threshold based on content quality distribution
            adjusted_threshold = self._adjust_similarity_threshold(similarities, min_similarity, content_type="heading")
            
            # Filter by adjusted similarity threshold and apply max limit
            result = []
            rank = 1
            for chunk in ranked_chunks:
                if chunk['similarity'] >= adjusted_threshold and len(result) < max_results:
                    result.append({
                        'document': chunk.get('document', 'unknown.pdf'),
                        'section_title': chunk['text'],
                        'importance_rank': rank,
                        'page_number': chunk['page']
                    })
                    rank += 1
            
            self.logger.info(f"Ranked {len(headings)} headings, returning {len(result)} above adjusted threshold {adjusted_threshold:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error ranking headings: {e}")
            return []
    
    def _rank_paragraphs(self, query: str, paragraphs: List[Dict[str, Any]], 
                        min_similarity: float, max_results: int) -> List[Dict[str, Any]]:
        """
        Rank paragraph chunks for subsection_analysis output format.
        Uses similarity threshold to include all relevant paragraphs.
        
        Args:
            query: Search query
            paragraphs: List of paragraph chunks to rank
            min_similarity: Minimum similarity threshold to include content
            max_results: Maximum number of results to prevent overwhelming output
            
        Returns:
            List of ranked paragraphs above similarity threshold in output schema format
        """
        if not paragraphs:
            return []
        
        try:
            texts = [chunk['text'] for chunk in paragraphs]
            
            # Generate embeddings for query and chunks
            query_embedding = self.embedding_model.encode_single(query)
            chunk_embeddings = self.embedding_model.encode(texts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                chunk_embeddings
            )[0]
            
            # Create ranked results with relevance boosting
            ranked_chunks = []
            for i, similarity in enumerate(similarities):
                chunk = paragraphs[i].copy()
                # Apply content relevance boost
                boost = self._calculate_content_relevance_boost(chunk['text'], query)
                boosted_similarity = min(1.0, similarity + boost)
                chunk['similarity'] = float(boosted_similarity)
                chunk['original_similarity'] = float(similarity)
                chunk['relevance_boost'] = float(boost)
                ranked_chunks.append(chunk)
            
            # Sort by similarity score (descending)
            ranked_chunks.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Intelligent filtering: adjust threshold based on content quality distribution - more lenient for paragraphs
            adjusted_threshold = self._adjust_similarity_threshold(similarities, min_similarity, content_type="paragraph")
            
            # Filter by adjusted similarity threshold and apply max limit
            result = []
            for chunk in ranked_chunks:
                if chunk['similarity'] >= adjusted_threshold and len(result) < max_results:
                    result.append({
                        'document': chunk.get('document', 'unknown.pdf'),
                        'refined_text': chunk['text'],
                        'page_number': chunk['page']
                    })
            
            self.logger.info(f"Ranked {len(paragraphs)} paragraphs, returning {len(result)} above adjusted threshold {adjusted_threshold:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error ranking paragraphs: {e}")
            return []
    
    def _adjust_similarity_threshold(self, similarities: np.ndarray, base_threshold: float, content_type: str = "heading") -> float:
        """
        Intelligently adjust similarity threshold based on content quality distribution.
        Paragraphs get more lenient thresholds since they naturally score lower than headings.
        
        Args:
            similarities: Array of similarity scores
            base_threshold: Base threshold to start from
            content_type: Type of content ("heading" or "paragraph")
            
        Returns:
            Adjusted threshold for high-quality content selection
        """
        if len(similarities) == 0:
            return base_threshold
        
        # Calculate statistics about similarity distribution
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        max_sim = np.max(similarities)
        median_sim = np.median(similarities)
        
        # Adjust base threshold for paragraph content (naturally scores lower)
        if content_type == "paragraph":
            base_threshold = max(0.25, base_threshold - 0.05)  # More lenient for paragraphs
        
        # More threshold adjustment
        if max_sim > 0.6:
            # If high-quality content available - be more selective
            # Use 75th percentile or mean + 0.5*std, whichever is higher
            percentile_75 = np.percentile(similarities, 75)
            statistical_threshold = mean_sim + 0.5 * std_sim
            adjusted = max(base_threshold, min(percentile_75, statistical_threshold))
        elif max_sim < 0.4:
            # If lower quality content - be more inclusive but still maintain standards
            adjusted = max(0.22, base_threshold - 0.05)
        else:
            # Normal case: use median-based approach for robustness
            if content_type == "paragraph":
                adjusted = max(base_threshold, median_sim) 
            else:
                adjusted = max(base_threshold, median_sim + 0.05)
        
        # Ensure reasonable bounds - more lenient for paragraphs
        if content_type == "paragraph":
            adjusted = max(0.22, min(0.6, adjusted))
        else:
            adjusted = max(0.25, min(0.65, adjusted))
        
        return adjusted
    
    def _calculate_content_relevance_boost(self, text: str, query: str) -> float:
        """
        Calculate additional relevance boost based on content quality indicators.
        """
        text_lower = text.lower()
        query_lower = query.lower()
        boost = 0.0
        
        # Extract key terms from query
        query_terms = [term.strip().lower() for term in query_lower.split(':')[-1].split()]
        
        # Boost for exact keyword matches
        for term in query_terms:
            if len(term) > 3 and term in text_lower:
                boost += 0.05
        
        # Boost for complete sentences (quality indicator)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count >= 1 and len(text) > 50:
            boost += 0.03
        
        # Boost for actionable content (contains verbs/instructions)
        action_words = ['create', 'make', 'fill', 'sign', 'complete', 'manage', 'use', 'add', 'select']
        action_count = sum(1 for word in action_words if word in text_lower)
        boost += min(0.05, action_count * 0.01)
        
        # Boost for specific domain terms
        if 'hr professional' in query_lower or 'human resources' in query_lower:
            hr_terms = ['employee', 'onboarding', 'compliance', 'workflow', 'process', 'documentation']
            hr_count = sum(1 for term in hr_terms if term in text_lower)
            boost += min(0.04, hr_count * 0.01)
        
        return min(0.15, boost)  # Limiting total boost at 0.15
    
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

            # Return top ranked results
            top_chunks = ranked_chunks[:top_k]
            
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

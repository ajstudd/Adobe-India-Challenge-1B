"""
PDF parsing module using PyMuPDF to extract headings and paragraphs.
Extracts text chunks with metadata including type, page number, and font size.
"""

import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any
from pathlib import Path

class PDFParser:
    """PDF parser to extract structured text chunks from PDF documents"""
    
    def __init__(self, heading_font_threshold: float = 12.0):
        """
        Initialize PDF parser.
        
        Args:
            heading_font_threshold: Font size threshold to identify headings
        """
        self.heading_font_threshold = heading_font_threshold
        self.logger = logging.getLogger(__name__)
    
    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parse PDF and extract text chunks with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        try:
            document_name = Path(pdf_path).name
            
            doc = fitz.open(pdf_path)
            self.logger.info(f"Opened PDF: {pdf_path} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks with formatting information
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]
                                
                                if text and len(text) > 3:  # Filter out very short text
                                    chunk_type = self._classify_text_type(text, font_size)
                                    
                                    chunk = {
                                        "type": chunk_type,
                                        "text": text,
                                        "page": page_num + 1,
                                        "font_size": round(font_size, 1),
                                        "document": document_name
                                    }
                                    chunks.append(chunk)
            
            doc.close()
            
            # Post-process chunks to merge fragments and clean up
            processed_chunks = self._post_process_chunks(chunks)
            
            self.logger.info(f"Extracted {len(processed_chunks)} chunks from {pdf_path}")
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {e}")
            return []
    
    def _classify_text_type(self, text: str, font_size: float) -> str:
        """
        Classify text as heading or paragraph based on font size and content.
        
        Args:
            text: Text content
            font_size: Font size of the text
            
        Returns:
            "heading" or "paragraph"
        """
        # Check if font size indicates heading (relaxed threshold)
        if font_size >= self.heading_font_threshold:
            return "heading"
        
        # Check for heading patterns (section titles, short lines)
        text_lower = text.lower().strip()
        heading_indicators = [
            "overview", "summary", "analysis", "report", "performance", 
            "trends", "strategy", "outlook", "metrics", "comparison",
            "assessment", "conclusion", "executive", "introduction",
            "chapter", "section", "methodology", "results", "discussion"
        ]
        
        # Short text that starts with heading indicators
        if len(text) < 150 and any(indicator in text_lower for indicator in heading_indicators):
            return "heading"
        
        # Text that looks like a title (mostly title case and short)
        if len(text) < 80 and (text.istitle() or text.isupper()):
            return "heading"
        
        # Lines that look like section headers (end with colon, no period)
        if len(text) < 100 and (text.endswith(':') or (not text.endswith('.') and text[0].isupper())):
            return "heading"
        
        return "paragraph"
    
    def _final_classify(self, text: str) -> str:
        """
        Final classification of text after merging.
        
        Args:
            text: Final merged text
            
        Returns:
            "heading" or "paragraph"
        """
        text_stripped = text.strip()
        
        # Very short text is likely a heading
        if len(text_stripped) < 80:
            return "heading"
        
        # Text with multiple sentences is likely a paragraph
        sentence_count = text_stripped.count('.') + text_stripped.count('!') + text_stripped.count('?')
        if sentence_count >= 2:
            return "paragraph"
        
        # Text starting with common heading words
        first_words = text_stripped.lower().split()[:3]
        heading_starters = ['executive', 'financial', 'market', 'revenue', 'investment', 'key', 'quarterly', 'annual']
        if any(starter in ' '.join(first_words) for starter in heading_starters):
            return "heading"
        
        return "paragraph"
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process chunks to merge fragments and remove duplicates.
        
        Args:
            chunks: Raw extracted chunks
            
        Returns:
            Processed and cleaned chunks
        """
        if not chunks:
            return []
        
        processed = []
        current_chunk = None
        
        for chunk in chunks:
            text = chunk["text"].strip()
            
            # Skip very short or empty text
            if len(text) < 5:
                continue
            
            # If this is a continuation of the previous chunk (same type, same page, similar font)
            if (current_chunk and 
                current_chunk["type"] == chunk["type"] and 
                current_chunk["page"] == chunk["page"] and
                current_chunk.get("document") == chunk.get("document") and
                abs(current_chunk["font_size"] - chunk["font_size"]) < 1.0 and
                chunk["type"] == "paragraph"):  # Only merge paragraphs, keep headings separate
                
                # Merge with current chunk if it's a paragraph
                current_chunk["text"] += " " + text
            else:
                # Start new chunk
                if current_chunk:
                    # Final classification check
                    final_text = current_chunk["text"].strip()
                    if len(final_text) >= 10:  # Only keep substantial chunks
                        # Re-classify based on final merged text
                        current_chunk["type"] = self._final_classify(final_text)
                        processed.append(current_chunk)
                
                current_chunk = chunk.copy()
        
        # Add the last chunk
        if current_chunk:
            final_text = current_chunk["text"].strip()
            if len(final_text) >= 10:
                current_chunk["type"] = self._final_classify(final_text)
                processed.append(current_chunk)
        
        # Remove duplicates and very similar chunks
        final_chunks = self._remove_duplicates(processed)
        
        return final_chunks
    
    def _remove_duplicates(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate and very similar chunks.
        
        Args:
            chunks: List of chunks to deduplicate
            
        Returns:
            Deduplicated chunks
        """
        unique_chunks = []
        seen_texts = set()
        
        for chunk in chunks:
            text = chunk["text"].strip()
            
            # Create a normalized version for comparison
            normalized = text.lower().replace(" ", "").replace("\n", "")
            
            # Skip if we've seen this exact text
            if normalized in seen_texts:
                continue
            
            # Skip if very similar to existing text
            is_similar = False
            for seen_text in seen_texts:
                if (len(normalized) > 20 and len(seen_text) > 20 and
                    (normalized in seen_text or seen_text in normalized)):
                    is_similar = True
                    break
            
            if not is_similar:
                seen_texts.add(normalized)
                unique_chunks.append(chunk)
        
        return unique_chunks

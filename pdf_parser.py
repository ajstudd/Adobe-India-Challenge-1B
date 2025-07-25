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
                
                page_chunks = self._extract_quality_chunks(page, page_num + 1, document_name)
                chunks.extend(page_chunks)
            
            doc.close()
            
            processed_chunks = self._enhance_chunk_quality(chunks)
            
            self.logger.info(f"Extracted {len(processed_chunks)} high-quality chunks from {pdf_path}")
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {e}")
            return []
    
    def _extract_quality_chunks(self, page, page_num: int, document_name: str) -> List[Dict[str, Any]]:
        """
        Extract high-quality, complete text chunks from a page.
        Focus on complete sentences and coherent paragraphs.
        """
        chunks = []
        
        # Get text with better paragraph structure
        text_dict = page.get_text("dict")
        
        current_paragraph = []
        current_font_size = 0
        paragraph_count = 0
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_font_sizes = []
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            line_text += text + " "
                            line_font_sizes.append(span["size"])
                    
                    if line_text.strip():
                        avg_font_size = sum(line_font_sizes) / len(line_font_sizes) if line_font_sizes else 12
                        
                        # Check if this is likely a new paragraph/section
                        if self._is_paragraph_break(line_text, avg_font_size, current_font_size):
                            # Save previous paragraph if it exists
                            if current_paragraph:
                                paragraph_text = " ".join(current_paragraph).strip()
                                if self._is_quality_content(paragraph_text):
                                    chunk_type = self._classify_text_type(paragraph_text, current_font_size)
                                    chunks.append({
                                        "type": chunk_type,
                                        "text": paragraph_text,
                                        "page": page_num,
                                        "font_size": round(current_font_size, 1),
                                        "document": document_name
                                    })
                            
                            current_paragraph = [line_text.strip()]
                            current_font_size = avg_font_size
                            paragraph_count += 1
                        else:
                            # Continue current paragraph
                            current_paragraph.append(line_text.strip())
                            current_font_size = avg_font_size
        
        if current_paragraph:
            paragraph_text = " ".join(current_paragraph).strip()
            if self._is_quality_content(paragraph_text):
                chunk_type = self._classify_text_type(paragraph_text, current_font_size)
                chunks.append({
                    "type": chunk_type,
                    "text": paragraph_text,
                    "page": page_num,
                    "font_size": round(current_font_size, 1),
                    "document": document_name
                })
        
        return chunks
    
    def _is_paragraph_break(self, line_text: str, font_size: float, prev_font_size: float) -> bool:
        """
        Determine if this line represents a paragraph break.
        """
        if prev_font_size == 0:
            return True
        
        if abs(font_size - prev_font_size) > 1.5:
            return True
        
        # Line starts with bullet points or numbers
        if line_text.startswith(('•', '▪', '1.', '2.', '3.', '4.', '5.', 'a)', 'b)', 'c)')):
            return True
        
        # Line is very short and might be a heading
        if len(line_text) < 80 and line_text.endswith((':')) and not line_text.endswith('.'):
            return True
        
        return False
    
    def _is_quality_content(self, text: str) -> bool:
        """
        Check if the text content is of sufficient quality for extraction.
        """
        text = text.strip()
        
        # Minimum length requirement
        if len(text) < 20:
            return False
        
        # Must contain some alphabetic characters
        if sum(c.isalpha() for c in text) < len(text) * 0.5:
            return False
        
        # Skip content that's mostly symbols or numbers
        if text.count('.') > 5 or text.count('_') > 5:
            return False
        
        # Check for meaningful content indicators
        meaningful_words = ['the', 'and', 'to', 'of', 'for', 'with', 'can', 'you', 'is', 'are', 'in', 'on', 'at']
        word_count = len(text.split())
        meaningful_count = sum(1 for word in text.lower().split() if word in meaningful_words)
        
        # Should have reasonable ratio of meaningful words
        if word_count > 5 and meaningful_count / word_count < 0.1:
            return False
        
        return True
    
    def _enhance_chunk_quality(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced post-processing to improve chunk quality and coherence.
        """
        if not chunks:
            return []
        
        enhanced_chunks = []
        
        # Remove duplicates and very similar content
        seen_texts = set()
        
        for chunk in chunks:
            text = chunk["text"].strip()
            
            # Skip if we've seen very similar content
            text_normalized = text.lower().replace(" ", "").replace("\n", "")
            if text_normalized in seen_texts:
                continue
            
            # Final quality check
            if self._is_quality_content(text) and len(text) >= 30:
                cleaned_text = self._clean_extracted_text(text)
                if cleaned_text and len(cleaned_text) >= 20:
                    chunk["text"] = cleaned_text
                    enhanced_chunks.append(chunk)
                    seen_texts.add(text_normalized)
        
        return enhanced_chunks
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and improve extracted text quality.
        """
        import re
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Fix common OCR issues
        text = text.replace("fi", "fi").replace("fl", "fl")
        
        # Remove trailing incomplete sentences if they don't end properly
        sentences = text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            text = '.'.join(sentences[:-1]) + '.'
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?', ':')):
            # If it looks like an incomplete sentence, don't add period
            words = text.split()
            if len(words) > 3 and text[-1].isalpha():
                text += '.'
        
        return text.strip()
    
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
                chunk["type"] == "paragraph"):
                
                current_chunk["text"] += " " + text
            else:
                if current_chunk:
                    final_text = current_chunk["text"].strip()
                    if len(final_text) >= 10:
                        # Re-classify based on final merged text
                        current_chunk["type"] = self._final_classify(final_text)
                        processed.append(current_chunk)
                
                current_chunk = chunk.copy()
        
        if current_chunk:
            final_text = current_chunk["text"].strip()
            if len(final_text) >= 10:
                current_chunk["type"] = self._final_classify(final_text)
                processed.append(current_chunk)
        
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

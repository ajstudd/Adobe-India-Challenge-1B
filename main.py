"""
Main entry point for Adobe Hackathon 1B - AI-powered PDF document analysis system.
Extracts relevant sections and paragraphs based on persona and task using semantic similarity.
"""

import json
import os
import sys
from pathlib import Path
import logging

from pdf_parser import PDFParser
from embedding_model import EmbeddingModel
from ranker import DocumentRanker
from utils import setup_logging, validate_input, ensure_directories

def main():
    """Main execution function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Adobe Hackathon 1B Document Analysis System")
    
    try:
        ensure_directories()
        
        input_path = Path("input/input.json")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        validate_input(input_data)
        
        persona = input_data.get('persona', {}).get('role', '')
        task = input_data.get('job_to_be_done', {}).get('task', '')
        documents = input_data.get('documents', [])
        challenge_info = input_data.get('challenge_info', {})
        
        logger.info(f"Processing {len(documents)} documents for persona: {persona}")
        logger.info(f"Challenge: {challenge_info.get('test_case_name', 'Unknown')}")
        
        pdf_parser = PDFParser()
        embedding_model = EmbeddingModel()
        ranker = DocumentRanker(embedding_model)
        
        all_chunks = []
        processed_docs = []
        for doc_info in documents:
            doc_path = Path("input/PDFs") / doc_info.get('filename', '')
            if doc_path.exists():
                logger.info(f"Processing document: {doc_path}")
                chunks = pdf_parser.parse_pdf(str(doc_path))
                all_chunks.extend(chunks)
                processed_docs.append(doc_info.get('filename', ''))
            else:
                logger.warning(f"Document not found: {doc_path}")
        
        if not all_chunks:
            logger.error("No valid documents found to process")
            logger.info("Available files in input/PDFs:")
            docs_dir = Path("input/PDFs")
            if docs_dir.exists():
                for file in docs_dir.iterdir():
                    logger.info(f"  - {file.name}")
            else:
                logger.info("  input/PDFs directory does not exist")
            raise ValueError("No valid documents found to process")
        
        query = f"{persona}: {task}"
        logger.info(f"Query: {query}")
        
        # Configuration - optimized for balanced output
        # Automatically determining optimal parameters for comprehensive coverage
        min_similarity = 0.32  # Slightly lower limit for better paragraph coverage
        max_sections = min(25, len(all_chunks) // 6)  
        max_subsections = min(15, len(all_chunks) // 8)
        
        # Ensure minimum useful output even for small document sets
        max_sections = max(5, max_sections)
        max_subsections = max(8, max_subsections)
        
        logger.info(f"Quality-optimized config: min_similarity={min_similarity}, max_sections={max_sections}, max_subsections={max_subsections}")
        logger.info(f"Total chunks extracted: {len(all_chunks)}")
        
        # Rank and extract relevant sections
        extracted_sections, subsection_analysis = ranker.rank_chunks(
            query, all_chunks, min_similarity, max_sections, max_subsections
        )
        
        from datetime import datetime
        output_data = {
            "metadata": {
                "input_documents": processed_docs,
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        output_path = Path("output/output.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results written to: {output_path}")
        logger.info(f"Extracted {len(extracted_sections)} sections and {len(subsection_analysis)} subsections")
        
        print(f"Processing complete! Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

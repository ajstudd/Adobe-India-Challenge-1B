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
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Adobe Hackathon 1B Document Analysis System")
    
    try:
        # Ensure required directories exist
        ensure_directories()
        
        # Load input configuration
        input_path = Path("inputs/input.json")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Validate input
        validate_input(input_data)
        
        # Extract persona and task
        persona = input_data.get('persona', '')
        task = input_data.get('task', '')
        documents = input_data.get('documents', [])
        
        logger.info(f"Processing {len(documents)} documents for persona: {persona}")
        
        # Initialize components
        pdf_parser = PDFParser()
        embedding_model = EmbeddingModel()
        ranker = DocumentRanker(embedding_model)
        
        # Parse all PDFs and extract chunks
        all_chunks = []
        processed_docs = []
        for doc_info in documents:
            doc_path = Path("inputs/docs") / doc_info.get('filename', '')
            if doc_path.exists():
                logger.info(f"Processing document: {doc_path}")
                chunks = pdf_parser.parse_pdf(str(doc_path))
                all_chunks.extend(chunks)
                processed_docs.append(doc_info.get('filename', ''))
            else:
                logger.warning(f"Document not found: {doc_path}")
        
        if not all_chunks:
            logger.error("No valid documents found to process")
            logger.info("Available files in inputs/docs:")
            docs_dir = Path("inputs/docs")
            if docs_dir.exists():
                for file in docs_dir.iterdir():
                    logger.info(f"  - {file.name}")
            else:
                logger.info("  inputs/docs directory does not exist")
            raise ValueError("No valid documents found to process")
        
        # Create query from persona and task
        query = f"{persona}: {task}"
        logger.info(f"Query: {query}")
        
        # Rank and extract relevant sections
        extracted_sections, subsection_analysis = ranker.rank_chunks(query, all_chunks)
        
        # Prepare output with metadata
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
        
        # Write output
        output_path = Path("outputs/output.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results written to: {output_path}")
        logger.info(f"Extracted {len(extracted_sections)} sections and {len(subsection_analysis)} subsections")
        
        print(f"✅ Processing complete! Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

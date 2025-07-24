"""
Utility functions for the Adobe Hackathon 1B project.
Includes logging setup, input validation, and helper functions.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

def setup_logging(level: int = logging.INFO, log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Optionally add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        "inputs",
        "inputs/docs",
        "outputs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_input(input_data: Dict[str, Any]) -> bool:
    """
    Validate input JSON structure.
    
    Args:
        input_data: Input data dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ['persona', 'task', 'documents']
    
    for field in required_fields:
        if field not in input_data:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(input_data['documents'], list):
        raise ValueError("'documents' field must be a list")
    
    if not input_data['documents']:
        raise ValueError("'documents' list cannot be empty")
    
    # Validate each document entry
    for i, doc in enumerate(input_data['documents']):
        if not isinstance(doc, dict):
            raise ValueError(f"Document {i} must be a dictionary")
        
        if 'filename' not in doc:
            raise ValueError(f"Document {i} missing 'filename' field")
    
    return True

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

def save_json(data: Dict[str, Any], file_path: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    # Ensure output directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    cleaned = " ".join(text.split())
    
    # Remove common artifacts
    cleaned = cleaned.replace('\x00', '')  # Remove null characters
    cleaned = cleaned.replace('\ufffd', '')  # Remove replacement characters
    
    return cleaned.strip()

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def create_sample_input(output_path: str = "inputs/input.json"):
    """
    Create a sample input.json file for testing.
    
    Args:
        output_path: Path to save sample input
    """
    sample_input = {
        "persona": "Financial Analyst",
        "task": "Analyze Q2 2024 financial performance and identify key trends",
        "documents": [
            {
                "filename": "company_report_q2_2024.pdf",
                "type": "financial_report"
            },
            {
                "filename": "market_analysis_2024.pdf",
                "type": "market_analysis"
            }
        ]
    }
    
    save_json(sample_input, output_path)
    print(f"Sample input created at: {output_path}")

def print_system_info():
    """Print system information for debugging"""
    import sys
    import platform
    
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script path: {os.path.abspath(__file__)}")
    print("=========================")

if __name__ == "__main__":
    # Create sample input for testing
    ensure_directories()
    create_sample_input()
    print_system_info()

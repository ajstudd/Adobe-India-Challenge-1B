# AI-Powered Document Analysis System

## Solution Overview

This system implements an advanced AI-powered document analysis engine for Adobe India Hackathon Challenge 1B. Our approach combines state-of-the-art semantic similarity algorithms with transformer-based embeddings to intelligently extract and rank relevant content from PDF documents based on user personas and tasks.

## Technical Approach

### Core Architecture

- **PDF Processing**: PyMuPDF (fitz) for robust text extraction and metadata preservation
- **Semantic Understanding**: sentence-transformers with all-MiniLM-L6-v2 model for generating contextual embeddings
- **Relevance Ranking**: Cosine similarity-based algorithms with adaptive thresholding
- **Content Classification**: Automatic detection of headings vs. paragraphs using font size analysis
- **Quality Assurance**: Statistical analysis for optimal content selection and duplicate removal

### Key Features

- **Intelligent Extraction**: Automatically classifies text into headings and paragraphs based on font characteristics
- **Semantic Similarity**: Uses pre-trained transformer models to understand context and meaning
- **Adaptive Thresholding**: Dynamically adjusts similarity thresholds based on content quality distribution
- **Scalable Processing**: Handles multiple PDF documents with automatic parameter optimization
- **Offline Operation**: Complete functionality without internet connectivity after initial setup

## Processing Pipeline

1. **Input Validation**: Validates JSON structure and file availability
2. **PDF Text Extraction**: Extracts text with font metadata for content classification
3. **Content Segmentation**: Separates headings from paragraphs using font size analysis
4. **Embedding Generation**: Creates semantic vectors for query and document content
5. **Similarity Computation**: Calculates cosine similarity between query and content
6. **Intelligent Ranking**: Applies adaptive thresholds and quality filters
7. **Output Generation**: Produces structured JSON with ranked relevant content

## Docker Build and Execution

### Prerequisites

- Docker installed and running
- Input files prepared in correct directory structure

### Build Instructions

**Step 1: Build Docker Image**

```bash
docker build --platform linux/amd64 -t mysolutionname.someidentifier .
```

**What happens during build:**

- Installs Python 3.10 and system dependencies
- Installs required Python packages (PyMuPDF, sentence-transformers, etc.)
- Downloads and embeds all-MiniLM-L6-v2 model for offline operation
- Creates necessary directory structure
- Sets up application environment

### Execution Instructions

**Step 2: Run Container**

```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none mysolutionname.someidentifier
```

**Parameters Explained:**

- `--rm`: Automatically removes container after execution
- `-v $(pwd)/input:/app/input:ro`: Mounts input directory as read-only
- `-v $(pwd)/output:/app/output`: Mounts output directory for results
- `--network none`: Ensures complete offline operation
- Final parameter: Your image name

## Directory Structure Requirements

The system expects the following directory structure:

```
input/
├── input.json          # Configuration file with persona, task, and document list
└── PDFs/              # Directory containing all PDF files to process
    ├── document1.pdf
    ├── document2.pdf
    └── ...

output/                # Results will be written here
└── output.json        # Generated analysis results
```

## Input File Specification

### input.json Format

```json
{
  "challenge_info": {
    "challenge_id": "string",
    "test_case_name": "string",
    "description": "string"
  },
  "persona": {
    "role": "string"
  },
  "job_to_be_done": {
    "task": "string"
  },
  "documents": [
    {
      "filename": "document1.pdf"
    },
    {
      "filename": "document2.pdf"
    }
  ]
}
```

### Required Fields

- **challenge_info**: Metadata about the test case
- **persona.role**: The role/profession of the user (e.g., "HR professional", "Developer")
- **job_to_be_done.task**: The specific task to accomplish (e.g., "Create fillable forms")
- **documents**: Array of PDF filenames that must exist in input/PDFs/ directory

## Output Specification

The system generates `output.json` in the output directory with the following structure:

```json
{
  "metadata": {
    "input_documents": ["document1.pdf", "document2.pdf"],
    "persona": "HR professional",
    "job_to_be_done": "Create fillable forms for onboarding",
    "processing_timestamp": "2025-01-XX-XX:XX:XX"
  },
  "extracted_sections": [
    {
      "document_name": "document1.pdf",
      "section_title": "Interactive Forms Section",
      "page_number": 5,
      "similarity_score": 0.754,
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document_name": "document1.pdf",
      "refined_text": "Complete paragraph text relevant to the task",
      "page_number": 5,
      "similarity_score": 0.621
    }
  ]
}
```

### Output Fields Explanation

- **metadata**: Processing information and input parameters
- **extracted_sections**: Ranked document sections (headings) most relevant to persona/task
- **subsection_analysis**: Ranked paragraphs most relevant to persona/task
- **similarity_score**: Semantic similarity score (0-1, higher = more relevant)
- **importance_rank**: Ranking by relevance (1 = most important)

## System Capabilities and Performance

### Processing Specifications

- **Runtime Environment**: Python 3.10+ with 2GB+ RAM for model operations
- **Document Capacity**: Handles up to 10 PDF documents simultaneously
- **Performance**: Completes analysis within 60 seconds for typical workloads
- **Content Support**: Multi-page documents with complex layouts and formatting
- **Quality Assurance**: Automatic duplicate detection, coherent paragraph reconstruction, and content validation

### Automated Intelligence Features

- **Adaptive Thresholding**: Automatically determines optimal similarity thresholds based on content quality
- **Dynamic Scaling**: Adjusts extraction limits based on document collection size
- **Statistical Analysis**: Uses distribution analysis to identify above-average relevant content
- **Content Classification**: Distinguishes between headings and paragraphs using font analysis
- **Quality Filtering**: Removes fragments, incomplete text, and low-quality content

## Offline Operation Guarantee

The solution operates completely offline after initial setup:

1. **Build Phase** (with internet): Downloads all-MiniLM-L6-v2 model and dependencies
2. **Execution Phase** (no internet): All processing uses pre-loaded models and local dependencies
3. **No External Calls**: Zero API calls or internet connectivity required during runtime
4. **Self-Contained**: All required resources embedded in Docker image

## Error Handling and Robustness

- **Input Validation**: Comprehensive validation of JSON structure and file availability
- **Graceful Degradation**: Continues processing if some documents are corrupted or missing
- **Detailed Logging**: Complete execution tracking for debugging and monitoring
- **Health Checks**: Built-in model verification and system status monitoring
- **Consistent Output**: Standardized JSON format regardless of input variations

This solution provides production-ready intelligent document analysis with enterprise-grade reliability, security, and performance characteristics suitable for automated evaluation environments.

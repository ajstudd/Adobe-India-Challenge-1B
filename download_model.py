"""
Model download script for offline usage.
Downloads the sentence-transformers model to local directory.
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

def download_model(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                  output_dir: str = './models/all-MiniLM-L6-v2'):
    """
    Download and save sentence transformer model for offline usage.
    
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model
    """
    print(f"Downloading model: {model_name}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download and save model
        model = SentenceTransformer(model_name)
        model.save(output_dir)
        
        print(f"✅ Model successfully saved to: {output_dir}")
        
        # Verify the model can be loaded
        print("Verifying model can be loaded...")
        test_model = SentenceTransformer(output_dir)
        test_embedding = test_model.encode(["This is a test sentence."])
        print(f"✅ Model verification successful. Embedding dimension: {test_embedding.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Download the model
    download_model()
    
    print("\n📋 Next steps:")
    print("1. The model is now available for offline usage")
    print("2. You can run the main application without internet access")
    print("3. For Docker deployment, the model will be copied into the container")

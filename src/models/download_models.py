"""
Model Downloader for KGRAG

Description:
  This script downloads and caches the necessary Hugging Face models for the KGRAG
  project. It should be run on a machine with internet access (e.g., a cluster
  login node) that has access to the shared cache directory.

  This ensures that when you run the main application on a compute node
  without internet access, the models are already available locally.

Usage:
  1. Ensure you have the required Python packages installed:
     pip install transformers sentence-transformers

  2. Set your cache directories (optional, but recommended on a cluster):
     export HF_HOME=/path/to/your/shared/.cache/huggingface
     export HF_HUB_CACHE=/path/to/your/shared/.cache/huggingface/hub

  3. Run the script:
     python download_models.py
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Models to download, taken from your config.yaml
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# --- Main Download Function ---
def download_models():
    """
    Downloads and caches the required LLM and embedding models from Hugging Face.
    """
    # Ensure the cache directory exists
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        print(f"Using Hugging Face cache directory: {hub_cache}")
        os.makedirs(hub_cache, exist_ok=True)
    else:
        print("HF_HUB_CACHE environment variable not set. Using default location.")

    print("\\n" + "="*50)
    print("Downloading models for KGRAG...")
    print("This may take a while depending on model size and internet speed.")
    print("="*50 + "\\n")

    # 1. Download LLM model and tokenizer
    print(f"--- Downloading LLM: {LLM_MODEL_NAME} ---")
    try:
        print("Downloading tokenizer...")
        AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        print("Downloading model...")
        AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        print(f"✅ Successfully downloaded and cached LLM: {LLM_MODEL_NAME}\\n")
    except Exception as e:
        print(f"❌ Error downloading LLM model '{LLM_MODEL_NAME}': {e}")
        print("Please check the model name and your internet connection.\\n")

    # 2. Download Embedding model
    print(f"--- Downloading Embedding Model: {EMBEDDING_MODEL_NAME} ---")
    try:
        print("Downloading model...")
        SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Successfully downloaded and cached Embedding Model: {EMBEDDING_MODEL_NAME}\\n")
    except Exception as e:
        print(f"❌ Error downloading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        print("Please check the model name and your internet connection.\\n")
    
    print("="*50)
    print("Model download process finished.")
    print("You should now be able to run the main application in an offline environment.")
    print("="*50)


if __name__ == "__main__":
    download_models() 
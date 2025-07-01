"""
Hugging Face Cache and Environment Diagnostic Tool
"""

import os
from pathlib import Path

# --- Configuration ---
REQUIRED_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-Embedding-0.6B"
]

def check_environment():
    """Prints the relevant Hugging Face environment variables."""
    print("--- 1. Checking Environment Variables ---")
    hf_home = os.environ.get("HF_HOME")
    hub_cache = os.environ.get("HF_HUB_CACHE")
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    offline = os.environ.get("HF_HUB_OFFLINE")

    print(f"HF_HOME: {hf_home or 'Not Set'}")
    print(f"HF_HUB_CACHE: {hub_cache or 'Not Set'}")
    print(f"HF_HUB_OFFLINE: {offline or 'Not Set'}")
    print(f"HUGGING_FACE_HUB_TOKEN: {'Set' if token else 'Not Set'}")

    if not hub_cache:
        default_home = hf_home or os.path.expanduser("~/.cache")
        hub_cache = str(Path(default_home) / "huggingface/hub")
        print(f"\\n[INFO] HF_HUB_CACHE is not set. Will check the default location:")
        print(f"  -> {hub_cache}")

    print("-" * 35 + "\\n")
    return hub_cache

def check_cache_directories(hub_cache: str):
    """Checks for the existence and permissions of the cache directory."""
    print("--- 2. Checking Cache Directory ---")
    if not hub_cache:
        print("[ERROR] Cannot check cache because cache path is not defined.")
        return

    cache_path = Path(hub_cache)
    print(f"Checking path: {cache_path}")

    if not cache_path.exists():
        print(f"[ERROR] The cache directory does not exist: {cache_path}")
    elif not os.access(cache_path, os.R_OK):
        print(f"[ERROR] The cache directory exists, but I don't have read permissions.")
    else:
        print("[SUCCESS] Cache directory exists and is readable.")
    
    print("-" * 35 + "\\n")

def check_models_in_cache(hub_cache: str):
    """Checks if the required model files are present in the cache."""
    print("--- 3. Checking for Models in Cache ---")
    if not hub_cache:
        print("[ERROR] Cannot check for models because cache path is not defined.")
        return

    all_found = True
    for model_name in REQUIRED_MODELS:
        # Hugging Face stores models by replacing '/' with '--'
        model_folder_name = f"models--{model_name.replace('/', '--')}"
        model_path = Path(hub_cache) / model_folder_name
        
        print(f"Looking for '{model_name}':")
        print(f"  -> at: {model_path}")

        if model_path.exists():
            config_file = model_path / "config.json"
            if config_file.exists():
                print(f"  [SUCCESS] Found model folder and config.json.")
            else:
                print(f"  [WARNING] Found model folder, but it seems incomplete (missing config.json).")
                all_found = False
        else:
            print(f"  [] Model folder not found.")
            all_found = False
        print()

    print("-" * 35 + "\\n")
    return all_found


if __name__ == "__main__":
    print("=" * 50)
    print("KGRAG Environment and Cache Diagnostic Tool")
    print("=" * 50 + "\\n")

    hub_cache_path = check_environment()
    if hub_cache_path:
        check_cache_directories(hub_cache_path)
        check_models_in_cache(hub_cache_path)

    print("\\n" + "=" * 50)
    print("Diagnostic finished. Compare this output with the environment on your login node.")
    print("=" * 50) 
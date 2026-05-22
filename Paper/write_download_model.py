"""
Create a highly visual, progress-tracking download script for BERT model weights.
This script will be written to D:\Open\College PC\Downloads\LLS_NEW\download_model.py.
"""
import os
BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

download_script = r'''"""
download_model.py - Visual BERT-base-uncased downloader with full progress bars.
"""
import os
import sys

# Disable symlink warnings on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("=" * 60)
print("BERT-base-uncased Direct Visual Downloader")
print("=" * 60)

try:
    from huggingface_hub import hf_hub_download
    from transformers import BertTokenizer
    import torch
except ImportError as e:
    print(f"Error: Missing required packages. Please run: pip install transformers huggingface_hub torch")
    sys.exit(1)

# Step 1: Pre-clean any locked or stuck incomplete files
import shutil
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--bert-base-uncased")
if os.path.exists(cache_dir):
    print("Pre-cleaning old cache files to ensure a clean download...")
    # Clean incomplete files
    blobs_dir = os.path.join(cache_dir, "blobs")
    if os.path.exists(blobs_dir):
        for f in os.listdir(blobs_dir):
            if f.endswith(".incomplete"):
                try:
                    os.remove(os.path.join(blobs_dir, f))
                    print(f"  Removed stuck incomplete file: {f}")
                except Exception as ex:
                    print(f"  Could not remove {f} (might be locked: {ex})")

print("\nStep 1: Downloading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer downloaded successfully!")

print("\nStep 2: Downloading Model weights (model.safetensors ~ 440MB)...")
print("This will show a real-time progress bar. Please wait...")

try:
    # Explicitly download model weights with tqdm progress bar enabled
    model_file = hf_hub_download(
        repo_id="bert-base-uncased",
        filename="model.safetensors",
        local_files_only=False
    )
    print("\n" + "=" * 60)
    print("SUCCESS: BERT Model weights successfully downloaded!")
    print(f"Weights saved at: {model_file}")
    print("=" * 60)
    
    # Verify the download can be loaded
    print("\nStep 3: Verifying the model can load successfully...")
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=18)
    print("Verification complete: Model loaded into memory successfully!")
    print("You are fully ready to run the web server!")
    
except Exception as e:
    print(f"\nDownload failed with error: {e}")
    print("If you are getting connection timeouts, please check your network connection.")
'''

with open(os.path.join(BASE, "download_model.py"), "w", encoding="utf-8") as f:
    f.write(download_script)
print(f"Written: {os.path.join(BASE, 'download_model.py')}")

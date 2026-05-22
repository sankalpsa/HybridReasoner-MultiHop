"""Patch bert_classifier.py to use absolute paths based on the file location."""
import os
BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

patch_script = r'''"""
Patch bert_classifier.py to resolve the model directory relative to its own file location,
ensuring robust offline model loading regardless of where Python is invoked.
"""
import os

filepath = r"D:\Open\College PC\Downloads\LLS_NEW\bert_classifier.py"

with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Make the model_dir default resolve to the absolute path of the directory this file lives in
target_line = 'def __init__(self, model_dir: str = "models/bert_kinship", device: str = None):'
replacement_line = 'def __init__(self, model_dir: str = None, device: str = None):\n        if model_dir is None:\n            # Resolve absolute path relative to the file directory\n            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "bert_kinship"))'

if target_line in code:
    code = code.replace(target_line, replacement_line)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print("bert_classifier.py patched successfully to use absolute pathing!")
else:
    print("Target line not found, might have been already updated.")
'''

with open(os.path.join(BASE, "patch_bert_path.py"), "w", encoding="utf-8") as f:
    f.write(patch_script)
print(f"Written: {os.path.join(BASE, 'patch_bert_path.py')}")

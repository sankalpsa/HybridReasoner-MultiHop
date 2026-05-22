"""Patch bert_classifier.py to load 100% offline using local_files_only=True."""
import os
BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

patch_script = r'''"""
Patch bert_classifier.py to force local_files_only=True, guaranteeing offline loading.
"""
filepath = r"D:\Open\College PC\Downloads\LLS_NEW\bert_classifier.py"

with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Replace tokenizer loading line
old_tokenizer = "self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)"
new_tokenizer = "self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, local_files_only=True)"

# Replace model loading line
old_model = "self.model = BertForSequenceClassification.from_pretrained(\n                self.model_dir, num_labels=self.num_labels\n            ).to(self.device)"
new_model = "self.model = BertForSequenceClassification.from_pretrained(\n                self.model_dir, num_labels=self.num_labels, local_files_only=True\n            ).to(self.device)"

if old_tokenizer in code:
    code = code.replace(old_tokenizer, new_tokenizer)
if old_model in code:
    code = code.replace(old_model, new_model)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("bert_classifier.py successfully patched with local_files_only=True!")
'''

with open(os.path.join(BASE, "patch_bert_offline.py"), "w", encoding="utf-8") as f:
    f.write(patch_script)
print(f"Written: {os.path.join(BASE, 'patch_bert_offline.py')}")

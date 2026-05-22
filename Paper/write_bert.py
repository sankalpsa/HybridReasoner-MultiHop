"""Write bert_classifier.py to D: drive."""
import os
BASE = r"D:\Open\College PC\Downloads\LLS_NEW"

code = r'''"""
bert_classifier.py - BERT-based Neural Reasoning Module

Fine-tuned BERT-base-uncased classifier for kinship relation prediction.
Input: [CLS] + Story + [SEP] + Query + [SEP]
Output: softmax over 18 kinship relations

References:
  - Paper Section 3.1, Equations 1-8, Table 1
  - BERT-base-uncased: 768-dim hidden, 12 heads, 12 layers
  - Classification head: R^768 -> R^18, dropout p=0.3
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# The 18 CLUTRR kinship relation labels (Table 2)
RELATION_LABELS = [
    "aunt", "brother", "daughter", "father",
    "father-in-law", "granddaughter", "grandfather", "grandmother",
    "grandson", "husband", "mother", "mother-in-law",
    "nephew", "niece", "sister", "son", "uncle", "wife",
]

REL2IDX = {r: i for i, r in enumerate(RELATION_LABELS)}
IDX2REL = {i: r for i, r in enumerate(RELATION_LABELS)}


class BERTKinshipClassifier:
    """
    BERT-based kinship relation classifier.

    Wraps HuggingFace transformers for:
      - Training on CLUTRR dataset
      - Inference: given (story, query) -> (relation, confidence)
    """

    def __init__(self, model_dir: str = "models/bert_kinship", device: str = None):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.max_length = 128  # Table 1
        self.num_labels = 18

    def load_or_init(self):
        """Load saved model or initialize from pretrained BERT."""
        from transformers import BertTokenizer, BertForSequenceClassification

        if os.path.exists(os.path.join(self.model_dir, "config.json")):
            logger.info(f"Loading fine-tuned model from {self.model_dir}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_dir, num_labels=self.num_labels
            ).to(self.device)
        else:
            logger.info("Initializing BERT-base-uncased with classification head")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=self.num_labels,
                hidden_dropout_prob=0.3,  # Paper: dropout p=0.3
            ).to(self.device)

        self.model.eval()

    def predict(self, story: str, query: str) -> Dict[str, Any]:
        """
        Predict kinship relation from story + query.

        Input format (Equation 1): [CLS] + Story + [SEP] + Query + [SEP]

        Returns:
            dict with 'relation', 'confidence', 'all_probs'
        """
        if self.model is None:
            self.load_or_init()

        # Tokenize: story = text_a, query = text_b (BERT pair format)
        encoding = self.tokenizer(
            story, query,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits  # shape: (1, 18)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_rel = IDX2REL[pred_idx]
        confidence = float(probs[pred_idx])

        return {
            "relation": pred_rel,
            "confidence": confidence,
            "all_probs": {IDX2REL[i]: float(probs[i]) for i in range(self.num_labels)},
        }

    def train_on_clutrr(
        self,
        csv_path: str = "clutrr_train.csv",
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
    ):
        """
        Fine-tune BERT on CLUTRR dataset (Table 1 hyperparameters).

        CSV must have columns: story/clean_story, query, target
        """
        import pandas as pd
        from transformers import get_linear_schedule_with_warmup

        if self.model is None:
            self.load_or_init()

        df = pd.read_csv(csv_path)
        logger.info(f"Training on {len(df)} examples from {csv_path}")

        # Prepare data
        stories, queries, labels = [], [], []
        for _, row in df.iterrows():
            story = str(row.get("clean_story", row.get("story", "")))
            query = str(row.get("query", ""))
            target = str(row.get("target", "")).strip().lower()
            if target in REL2IDX and story and query:
                stories.append(story)
                queries.append(query)
                labels.append(REL2IDX[target])

        logger.info(f"Usable examples: {len(stories)}")
        if len(stories) < 10:
            logger.error("Too few usable examples. Check CSV columns.")
            return

        # Tokenize all
        encodings = self.tokenizer(
            stories, queries,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_tensor = torch.tensor(labels, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            encodings.get("token_type_ids", torch.zeros_like(encodings["input_ids"])),
            label_tensor,
        )

        # Split 90/10
        n_val = max(1, int(0.1 * len(dataset)))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Optimizer (AdamW per paper)
        from transformers import AdamW
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            total_loss = 0
            self.model.train()
            for batch in train_loader:
                ids, mask, tids, labs = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                out = self.model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
                loss = loss_fn(out.logits, labs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            # Validation
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids, mask, tids, labs = [b.to(self.device) for b in batch]
                    out = self.model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
                    preds = out.logits.argmax(dim=-1)
                    correct += (preds == labs).sum().item()
                    total += labs.size(0)

            val_acc = correct / max(total, 1)
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print("Early stopping (patience=3)")
                    break

        print(f"Training complete. Best val_acc={best_val_acc:.4f}")
        self.model.eval()

    def save(self):
        """Save model and tokenizer."""
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        logger.info(f"Model saved to {self.model_dir}")

    def neural_predict_fn(self, narrative: str, subject: str, obj: str) -> Dict[str, Any]:
        """
        Wrapper matching the signature expected by orchestrator.hybrid_select().
        Constructs a query string from subject and object, then predicts.
        """
        query = f"What is the relationship between {subject} and {obj}?"
        return self.predict(narrative, query)


# Singleton for use by the Flask app
_classifier = None

def get_classifier(model_dir: str = "models/bert_kinship") -> BERTKinshipClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BERTKinshipClassifier(model_dir=model_dir)
        try:
            _classifier.load_or_init()
        except Exception as e:
            logger.warning(f"Could not load BERT model: {e}")
    return _classifier


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        csv = sys.argv[2] if len(sys.argv) > 2 else "clutrr_train.csv"
        clf = BERTKinshipClassifier()
        clf.load_or_init()
        clf.train_on_clutrr(csv)
    else:
        print("Usage: python bert_classifier.py train [clutrr_train.csv]")
        print("  Or import and use BERTKinshipClassifier directly.")
'''

with open(os.path.join(BASE, "bert_classifier.py"), "w", encoding="utf-8") as f:
    f.write(code)
print(f"Written bert_classifier.py ({len(code)} bytes)")

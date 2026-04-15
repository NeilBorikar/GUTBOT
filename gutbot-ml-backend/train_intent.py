import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from torch import nn
import argparse
import random
import logging

# ====================== Logging Config ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ====================== Dataset Class ======================
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.label2id[self.data[idx]["label"]]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label)
        }

# ====================== Training Function ======================
def train_intent_model(data_path, model_dir, epochs=75, batch_size=8, lr=2e-5, val_split=0.12, seed=42):
    # ---- Load Data ----
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # ---- Reproducibility ----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    labels = sorted(list(set(item["label"] for item in all_data)))
    label2id = {l: i for i, l in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # ---- Train / Val Split ----
    train_items, val_items = train_test_split(
        all_data, test_size=val_split, random_state=seed, stratify=[x['label'] for x in all_data]
    )

    train_dataset = IntentDataset(train_items, tokenizer, label2id)
    val_dataset = IntentDataset(val_items, tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- Model Setup ----
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- Weighted Loss ----
    class_counts = {label: sum(1 for d in all_data if d['label'] == label) for label in labels}
    weights = [1.0 / class_counts[label] for label in labels]
    weights_tensor = torch.tensor(weights).to(device)
    loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    # ---- Early Stopping ----
    patience = 5
    no_improve = 0
    best_val_acc = -1.0
    best_epoch = -1

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Starting training for up to %d epochs...", epochs)

    # ====================== TRAIN LOOP ======================
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"🧠 Training Epoch {epoch}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_fct(logits, inputs["labels"])

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # ====================== VALIDATION ======================
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for vb in val_loader:
                vinputs = {k: v.to(device) for k, v in vb.items()}
                labels_tensor = vinputs.pop("labels")
                outputs = model(**vinputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                correct += (preds == labels_tensor.cpu().numpy()).sum()
                total += labels_tensor.size(0)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels_tensor.cpu().numpy().tolist())

        val_acc = correct / total if total > 0 else 0.0
        logger.info(f"📊 Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ---- Generate & Save Classification Report ----
        report = classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True
        )
        with open(model_dir / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        # ====================== EARLY STOPPING ======================
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0

            # Save best model
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            with open(model_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump({"label2id": label2id, "id2label": {v: k for k, v in label2id.items()}}, f, indent=4)

            torch.save(model.state_dict(), model_dir / "best_model_state.pt")
            logger.info(f"✅ New best model saved (Epoch {epoch}) | Val Acc: {val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.warning(f"⛔ No improvement for {patience} epochs. Early stopping at epoch {epoch}.")
                break

    logger.info(f"🏁 Training complete. Best epoch: {best_epoch} | Best Val Acc: {best_val_acc:.4f}")

    # Return trained model and tokenizer
    best_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    best_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return best_model, best_tokenizer

# ====================== MAIN ENTRY ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="C:/Users/NEIL/projects/public_health_chatbot/gutbot-ml-backend/training_data/intents.json")
    parser.add_argument("--model_dir", type=str, default="C:/Users/NEIL/projects/public_health_chatbot/gutbot-ml-backend/models/intent_classifier")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val_split", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_intent_model(
        data_path=args.data_path,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed
    )

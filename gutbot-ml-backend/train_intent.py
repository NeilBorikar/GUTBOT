import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import json
from pathlib import Path

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

def train_intent_model(data_path, model_dir, epochs=3, batch_size=8, lr=2e-5):
    # Load training data
    with open(data_path, "r") as f:
        data = json.load(f)

    labels = sorted(list(set(item["label"] for item in data)))
    label2id = {l: i for i, l in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = IntentDataset(data, tokenizer, label2id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels)
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * epochs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # ✅ Save model + tokenizer
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # ✅ Save label mappings
    with open(model_dir / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": {v: k for k, v in label2id.items()}}, f, indent=4)

    print(f"✅ Model and tokenizer saved at {model_dir}")

    return model, tokenizer

if __name__ == "__main__":
    train_intent_model(
        data_path="C:/Users/NEIL/projects/public_health_chatbot/training_data/intents.json",  # ✅ full path
        model_dir="C:/Users/NEIL/projects/public_health_chatbot/Backend/models/intent_classifier",  # ✅ full path
        epochs=5
    )

import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_DIR = Path("../medical_data")
INDEX_DIR = Path("./models/medical_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_medical_data():
    records = []
    for category in ["diseases", "symptoms", "prevention", "treatments", "medications"]:
        folder = DATA_DIR / category
        if not folder.exists():
            continue
        for file in folder.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    name = data.get("name", file.stem)
                    description = data.get("description", "")
                    
                    # Store unified record
                    records.append({
                        "category": category,
                        "name": name,
                        "description": description,
                        "symptoms": data.get("symptoms", []),
                        "prevention": data.get("prevention", []),
                        "treatments": data.get("treatments", []),
                        "medications": data.get("medications", []),
                        "file": str(file)
                    })
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return records

def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    records = load_medical_data()

    if not records:
        print("⚠️ No medical data found in ../medical_data. Please add JSON files.")
        return

    texts = [f"{r['name']} {r['description']}" for r in records if r['name'] or r['description']]
    embeddings = model.encode(texts, convert_to_numpy=True)

    if embeddings.shape[0] == 0:
        print("⚠️ No valid embeddings generated. Check your medical_data JSON files.")
        return

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "medical.index"))
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"✅ Medical KB index built with {len(records)} entries!")


if __name__ == "__main__":
    build_index()

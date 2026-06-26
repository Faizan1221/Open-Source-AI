"""
upload_kb.py — loads knowledge_base.csv into ChromaDB.

Usage:
    python upload_kb.py --file knowledge_base.csv
    railway run python upload_kb.py --file knowledge_base.csv
"""

import os
import csv
import argparse
import chromadb


def load_csv(filepath: str):
    pairs = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get('question', '').strip()
            a = row.get('answer', '').strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def upload(filepath: str):
    CHROMA_PATH     = os.environ.get('CHROMA_PATH', './chroma_db')
    COLLECTION_NAME = os.environ.get('COLLECTION_NAME', 'company_kb')

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    pairs = load_csv(filepath)
    if not pairs:
        print("No valid Q&A pairs found. Check column names: 'question' and 'answer'.")
        return

    existing = collection.count()
    if existing > 0:
        print(f"Clearing {existing} existing entries...")
        all_ids = collection.get()['ids']
        collection.delete(ids=all_ids)

    documents = []
    metadatas = []
    ids       = []

    for i, (question, answer) in enumerate(pairs):
        documents.append(answer)
        metadatas.append({"question": question})
        ids.append(f"qa_{i}")

    batch_size = 50
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )
        print(f"Uploaded {min(end, len(documents))}/{len(documents)} entries...")

    print(f"\nDone! {len(pairs)} Q&A pairs uploaded to ChromaDB collection '{COLLECTION_NAME}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upload Q&A CSV to ChromaDB')
    parser.add_argument('--file', required=True, help='Path to knowledge_base.csv')
    args = parser.parse_args()
    upload(args.file)
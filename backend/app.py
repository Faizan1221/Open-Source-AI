import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "chromadb==0.5.3", "sentence-transformers==2.7.0"])
import os
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')
CORS(app, origins=ALLOWED_ORIGINS.split(','))

CHROMA_PATH      = os.environ.get('CHROMA_PATH', './chroma_db')
COLLECTION_NAME  = os.environ.get('COLLECTION_NAME', 'company_kb')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.70'))

client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

@app.route('/health', methods=['GET'])
def health():
    count = collection.count()
    return jsonify({"status": "ok", "entries": count})

@app.route('/ask', methods=['POST'])
def ask():
    data     = request.get_json(force=True)
    question = data.get('question', '').strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    count = collection.count()
    if count == 0:
        return jsonify({
            "answer"  : "My knowledge base is still being loaded. Please try again shortly.",
            "in_scope": False,
            "score"   : 0.0
        })

    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    if not results['documents'] or not results['documents'][0]:
        return jsonify({"answer": "I don't have information on that topic yet.", "in_scope": False, "score": 0.0})

    distance = results['distances'][0][0]
    score    = round(1 - distance, 4)
    answer   = results['documents'][0][0]

    if score < SIMILARITY_THRESHOLD:
        return jsonify({
            "answer"  : "I'm not sure about that one. Please contact HR or your manager for more details.",
            "in_scope": False,
            "score"   : score
        })

    return jsonify({"answer": answer, "in_scope": True, "score": score})

@app.route('/entries', methods=['GET'])
def entries():
    count = collection.count()
    all_items = collection.get(limit=count)
    pairs = []
    for i, doc in enumerate(all_items['documents']):
        meta = all_items['metadatas'][i] if all_items['metadatas'] else {}
        pairs.append({"question": meta.get('question', ''), "answer": doc})
    return jsonify({"count": count, "entries": pairs})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
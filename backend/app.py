import os
import csv
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')
CORS(app, origins=ALLOWED_ORIGINS.split(','))

SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.70'))
KB_PATH = os.environ.get('KB_PATH', './knowledge_base.csv')

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory knowledge base
questions = []
answers = []
embeddings = []

def load_knowledge_base():
    global questions, answers, embeddings
    if not os.path.exists(KB_PATH):
        print(f"Knowledge base not found at {KB_PATH}")
        return
    pairs = []
    with open(KB_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get('question', '').strip()
            a = row.get('answer', '').strip()
            if q and a:
                pairs.append((q, a))
    if not pairs:
        print("No Q&A pairs found in CSV.")
        return
    questions = [p[0] for p in pairs]
    answers   = [p[1] for p in pairs]
    embeddings = model.encode(questions, normalize_embeddings=True)
    print(f"Loaded {len(pairs)} Q&A pairs into memory.")

def cosine_similarity(query_emb, corpus_emb):
    return np.dot(corpus_emb, query_emb)

# Load on startup
load_knowledge_base()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "entries": len(questions)})

@app.route('/ask', methods=['POST'])
def ask():
    data     = request.get_json(force=True)
    question = data.get('question', '').strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not questions:
        return jsonify({
            "answer"  : "My knowledge base is still being loaded. Please try again shortly.",
            "in_scope": False,
            "score"   : 0.0
        })

    query_emb = model.encode([question], normalize_embeddings=True)[0]
    scores    = cosine_similarity(query_emb, embeddings)
    best_idx  = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return jsonify({
            "answer"  : "I'm not sure about that one. Please contact HR or your manager for more details.",
            "in_scope": False,
            "score"   : round(best_score, 4)
        })

    return jsonify({
        "answer"  : answers[best_idx],
        "in_scope": True,
        "score"   : round(best_score, 4)
    })

@app.route('/entries', methods=['GET'])
def entries():
    pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
    return jsonify({"count": len(pairs), "entries": pairs})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
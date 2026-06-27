import os
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')
CORS(app, origins=ALLOWED_ORIGINS.split(','))

SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.15'))
KB_PATH = os.environ.get('KB_PATH', './knowledge_base.csv')

questions = []
answers = []
vectorizer = None
matrix = None

def load_knowledge_base():
    global questions, answers, vectorizer, matrix
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
        print("No Q&A pairs found.")
        return
    questions = [p[0] for p in pairs]
    answers   = [p[1] for p in pairs]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(questions)
    print(f"Loaded {len(pairs)} Q&A pairs.")

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
        return jsonify({"answer": "Knowledge base not loaded.", "in_scope": False, "score": 0.0})
    query_vec = vectorizer.transform([question])
    scores    = cosine_similarity(query_vec, matrix).flatten()
    best_idx  = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score < SIMILARITY_THRESHOLD:
        return jsonify({"answer": "I'm not sure about that. Please contact HR or your manager.", "in_scope": False, "score": round(best_score, 4)})
    return jsonify({"answer": answers[best_idx], "in_scope": True, "score": round(best_score, 4)})

@app.route('/entries', methods=['GET'])
def entries():
    pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
    return jsonify({"count": len(pairs), "entries": pairs})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
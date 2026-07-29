"""
app.py
------
Interface web pour générer du texte avec le modèle de langage entraîné.

Lancement (après `python train.py`) :
    python app.py
"""

import os
from flask import Flask, request, jsonify, render_template

from model import LSTMLM
from generate import generate
from bpe import encode as bpe_encode, decode as bpe_decode
from retrieval import load_documents_from_dir, BM25Index
from embeddings import LSTMEmbeddingIndex, HybridIndex
from agent import ReActAgent, SearchTool
from memory import get_memory

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.npz")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")

model = None
stoi = None
itos = None
merges = None
model_ready = False

if os.path.exists(MODEL_PATH):
    model, stoi, itos, merges = LSTMLM.load(MODEL_PATH)
    model_ready = True
    print(f"[app] Modèle chargé : vocabulaire BPE de {model.vocab_size} tokens, "
          f"contexte de {model.block_size} tokens.")
else:
    print(f"[app] Aucun modèle trouvé à {MODEL_PATH}. Lance `python train.py` d'abord.")

# Index documentaire pour /api/ask : un fichier .txt = un document, dans
# data/documents/. Optionnel — si le dossier est vide ou absent, /api/ask
# renverra une erreur explicite plutôt que de planter.
CHUNK_SIZE = 200   # tokens par tranche
CHUNK_OVERLAP = 50  # tokens de chevauchement entre deux tranches consécutives

# Si un modèle est déjà entraîné, on découpe avec ses VRAIS tokens BPE
# (plus fidèle) ; sinon on retombe sur un découpage par mots (voir
# retrieval.chunk_text). Dans les deux cas, la taille des tranches
# reste comparable, ce qui compte pour BM25 (voir retrieval.py).
_chunk_kwargs = {}
if model_ready:
    _chunk_kwargs = {
        "encode_fn": lambda text: bpe_encode(text, merges, stoi),
        "decode_fn": lambda ids: bpe_decode(ids, itos),
    }

passages = load_documents_from_dir(DOCS_DIR, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                                    **_chunk_kwargs)

doc_index = None
if passages:
    lexical_index = BM25Index(passages)
    doc_index = lexical_index
    print(f"[app] Index BM25 : {len(passages)} tranches ({CHUNK_SIZE} tokens, "
          f"chevauchement {CHUNK_OVERLAP}) depuis {DOCS_DIR}.")

    if model_ready:
        try:
            semantic_index = LSTMEmbeddingIndex(passages, model, stoi, itos, merges)
            doc_index = HybridIndex(lexical_index, semantic_index)
            print("[app] Index sémantique LSTM construit — recherche hybride "
                  "BM25 + embeddings activée (fusion par rang, voir embeddings.py).")
        except Exception as exc:  # pragma: no cover - garde-fou, ne doit pas bloquer l'app
            print(f"[app] Index sémantique indisponible ({exc}) — repli sur BM25 seul.")
else:
    print(f"[app] Aucun document trouvé dans {DOCS_DIR} (mets des .txt là pour activer /api/ask).")


def _answer_from_results(question: str, results: list, mode: str, **kwargs) -> tuple:
    """Formule la réponse finale à partir des passages retenus par
    l'agent. C'est exactement l'ancienne logique extractif/génératif de
    /api/ask, juste déplacée ici pour servir d'`answerer` à l'agent —
    l'agent ne sait pas COMMENT on répond, seulement QUAND il a assez
    d'information pour le faire."""
    if mode == "generative":
        if not model_ready:
            raise ValueError("Mode génératif demandé mais aucun modèle entraîné. "
                              "Lance train.py, ou utilise mode='extractive'.")
        top_score = results[0]["score"]
        grounded_results = [r for r in results if r["score"] >= 0.6 * top_score]
        context = "\n".join(r["text"] for r in grounded_results)

        length = max(5, min(int(kwargs.get("length", 80)), 300))
        temperature = max(0.05, min(float(kwargs.get("temperature", 0.6)), 2.5))
        prompt = f"Contexte : {context}\nQuestion : {question}\nRéponse :"
        answer = generate(model, merges, stoi, itos, prompt=prompt, length=length,
                           temperature=temperature, top_k=30, top_p=0.9,
                           context_bias_text=context, context_bias_strength=2.5)
        note = ("Réponse générée par le LSTM à partir du contexte ci-dessous — "
                "ce petit modèle n'a jamais été entraîné sur des paires question/réponse, "
                "donc rien ne garantit qu'il s'appuie fidèlement sur le contexte plutôt "
                "que d'inventer. Vérifie toujours avec les sources.")
    else:
        answer = results[0]["text"]
        note = "Réponse extraite directement du document le plus pertinent (aucune génération)."
    return answer, note


search_tool = SearchTool(doc_index)
agent = ReActAgent(tool=search_tool, answerer=_answer_from_results)


@app.route("/")
def index():
    return render_template("index.html", model_ready=model_ready, docs_ready=doc_index is not None)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    if not model_ready:
        return jsonify({"error": "Aucun modèle entraîné. Lance train.py d'abord."}), 400

    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    length = int(data.get("length", 60))
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 0))
    top_p = float(data.get("top_p", 1.0))

    length = max(5, min(length, 300))
    temperature = max(0.05, min(temperature, 2.5))
    top_k = max(0, min(top_k, 200))
    top_p = max(0.05, min(top_p, 1.0))

    if not prompt:
        prompt = "Il était"

    text = generate(model, merges, stoi, itos, prompt=prompt, length=length,
                     temperature=temperature, top_k=top_k, top_p=top_p)
    return jsonify({"text": text})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    if doc_index is None:
        return jsonify({
            "error": f"Aucun document indexé. Ajoute des fichiers .txt dans {DOCS_DIR}/ et relance l'app."
        }), 400

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    mode = data.get("mode", "extractive")  # "extractive" ou "generative"
    top_k = max(1, min(int(data.get("top_k", 3)), 10))
    min_score = max(0.0, min(float(data.get("min_score", 0.12)), 1.0))

    if not question:
        return jsonify({"error": "Question vide."}), 400
    if mode == "generative" and not model_ready:
        return jsonify({"error": "Mode génératif demandé mais aucun modèle entraîné. "
                                  "Lance train.py, ou utilise mode='extractive'."}), 400

    # Mémoire explicite : le client envoie le conversation_id reçu à son
    # premier échange (absent au tout premier appel -> on en crée un).
    memory = get_memory(conversation_id=data.get("conversation_id"))

    agent.min_score = min_score  # les curseurs de l'UI restent branchés
    # L'agent doit voir la mémoire telle qu'elle était AVANT cette
    # question (pour résoudre 'et ça ?' par rapport au tour précédent) —
    # on n'enregistre donc la question courante qu'après coup.
    result = agent.run(question, memory=memory, mode=mode, top_k=top_k,
                        length=data.get("length", 80), temperature=data.get("temperature", 0.6))

    memory.add("user", question)
    if result["answer"] is not None:
        memory.add("assistant", result["answer"])

    return jsonify({
        "answer": result["answer"],
        "mode": mode,
        "sources": result["sources"],
        "note": result["note"],
        "trace": result["trace"],              # Thought/Action/Observation/Final Answer
        "conversation_id": memory.conversation_id,
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Repart d'une conversation vierge (bouton 'Nouvelle conversation'
    côté client, par exemple)."""
    data = request.get_json(force=True)
    conversation_id = data.get("conversation_id")
    if conversation_id:
        get_memory(conversation_id=conversation_id).clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("[app] Ouvre http://127.0.0.1:5002 dans ton navigateur.")
    app.run(host="127.0.0.1", port=5002, debug=False)

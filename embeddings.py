"""
embeddings.py
-------------
Recherche sémantique par embeddings, en restant dans l'esprit "100%
NumPy, pas de gros framework externe" du reste du projet.

Deux options possibles pour obtenir un vecteur par passage :

  A) LSTMEmbeddingIndex (implémentée ci-dessous, ZÉRO dépendance
     nouvelle) : on réutilise le LSTM DÉJÀ entraîné par train.py.
     Ce n'est pas un modèle d'embeddings à proprement parler (il n'a
     jamais été entraîné pour ça, juste à prédire le mot suivant),
     mais son état caché résume quand même, approximativement, "de
     quoi parle" une séquence de tokens — donc en moyennant cet état
     caché sur tous les tokens d'un passage, on obtient un vecteur
     exploitable pour une similarité cosinus. Avantage : aucune
     dépendance supplémentaire, cohérent avec la philosophie du repo.
     Limite honnête : la qualité sémantique dépend ENTIÈREMENT de la
     qualité du LSTM entraîné (avec un petit corpus et peu d'epochs,
     ne pas s'attendre à des miracles — voir la note renvoyée par
     `LSTMEmbeddingIndex.search`).

  B) SentenceTransformerEmbeddingIndex (squelette fourni, dépendance
     OPTIONNELLE) : si tu veux une vraie qualité sémantique, la
     suggestion `sentence-transformers` de l'énoncé est la bonne
     piste — un petit modèle comme `all-MiniLM-L6-v2` (~80 Mo, tourne
     bien sur CPU) donnera des embeddings nettement meilleurs que le
     LSTM maison. Elle n'est PAS active par défaut (pas dans
     requirements.txt) pour ne rien casser chez qui ne l'installe pas ;
     voir la classe plus bas pour l'activer.

Les deux classes exposent `search(query, top_k) -> list[dict(doc_id,
chunk_id, text, score)]`, comme BM25Index/TfidfIndex — interchangeables
du point de vue de agent.SearchTool.

HybridIndex combine un index lexical (BM25) et un index sémantique
(embeddings) par Reciprocal Rank Fusion (RRF) plutôt que par une somme
pondérée des scores : un score BM25 et une similarité cosinus ne
vivent pas sur la même échelle (l'un est peut-être 8.3, l'autre 0.62),
les additionner directement n'a pas de sens sans normalisation
arbitraire. RRF ne regarde que le RANG de chaque passage dans chaque
index, ce qui est déjà comparable — c'est la méthode standard utilisée
par la plupart des systèmes de recherche hybride en production.
"""

import numpy as np

from bpe import encode as bpe_encode


class LSTMEmbeddingIndex:
    """Embeddings "gratuits" tirés de l'état caché du LSTM déjà
    entraîné (voir docstring du module). Construit une seule fois
    (à l'import de l'app), comme BM25Index/TfidfIndex."""

    def __init__(self, passages: list, model, stoi: dict, itos: dict, merges: list):
        self.passages = passages
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.merges = merges
        self.doc_vectors = None  # (n_docs, hidden_dim), normalisés
        self._build()

    def _embed(self, text: str) -> np.ndarray:
        """Moyenne de l'état caché (dernière couche) du LSTM sur tous
        les tokens du texte — une forme de 'mean pooling', la façon la
        plus simple de réduire une séquence de vecteurs à un seul
        vecteur de taille fixe (utilisée aussi par des vrais modèles
        d'embeddings comme les premières versions de Sentence-BERT)."""
        ids = bpe_encode(text, self.merges, self.stoi)
        if not ids:
            return np.zeros(self.model.hidden_dim, dtype=np.float64)
        h, c = self.model.init_state(batch_size=1)
        hidden_states = []
        for tok_id in ids:
            probs, h, c = self.model.step(np.array([tok_id]), h, c)
            hidden_states.append(h[-1][0])  # dernière couche, seul élément du batch
        vec = np.mean(hidden_states, axis=0).astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _build(self):
        if not self.passages:
            self.doc_vectors = np.zeros((0, self.model.hidden_dim))
            return
        self.doc_vectors = np.stack([self._embed(p["text"]) for p in self.passages])

    def search(self, query: str, top_k: int = 3) -> list:
        if self.doc_vectors is None or len(self.passages) == 0:
            return []
        qvec = self._embed(query)
        scores = self.doc_vectors @ qvec  # cosinus, puisque tout est déjà normalisé
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"doc_id": self.passages[i]["doc_id"], "chunk_id": self.passages[i]["chunk_id"],
             "text": self.passages[i]["text"], "score": float(scores[i])}
            for i in order
        ]


class SentenceTransformerEmbeddingIndex:
    """Squelette pour une vraie recherche sémantique via un petit
    modèle d'embeddings ONNX/sentence-transformers, tel que suggéré
    dans l'énoncé. Dépendance optionnelle : n'installe/n'importe
    `sentence-transformers` que si cette classe est réellement
    utilisée, pour ne rien casser chez qui n'en a pas besoin.

        pip install sentence-transformers

    Utilisation (dans app.py, à la place de LSTMEmbeddingIndex) :

        from embeddings import SentenceTransformerEmbeddingIndex
        semantic_index = SentenceTransformerEmbeddingIndex(
            passages, model_name="all-MiniLM-L6-v2")
    """

    def __init__(self, passages: list, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers n'est pas installé. "
                "Lance `pip install sentence-transformers` pour utiliser "
                "SentenceTransformerEmbeddingIndex, ou utilise LSTMEmbeddingIndex "
                "(zéro dépendance, voir embeddings.py)."
            ) from exc
        self.passages = passages
        self._encoder = SentenceTransformer(model_name)
        self.doc_vectors = None
        self._build()

    def _build(self):
        if not self.passages:
            self.doc_vectors = np.zeros((0, self._encoder.get_sentence_embedding_dimension()))
            return
        vecs = self._encoder.encode([p["text"] for p in self.passages], normalize_embeddings=True)
        self.doc_vectors = np.asarray(vecs)

    def search(self, query: str, top_k: int = 3) -> list:
        if self.doc_vectors is None or len(self.passages) == 0:
            return []
        qvec = self._encoder.encode([query], normalize_embeddings=True)[0]
        scores = self.doc_vectors @ qvec
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"doc_id": self.passages[i]["doc_id"], "chunk_id": self.passages[i]["chunk_id"],
             "text": self.passages[i]["text"], "score": float(scores[i])}
            for i in order
        ]


class HybridIndex:
    """Combine un index lexical et un index sémantique par Reciprocal
    Rank Fusion : score(passage) = somme sur chaque index de
    1 / (k_rrf + rang du passage dans cet index). k_rrf=60 est la
    valeur standard de la littérature (Cormack et al., 2009) — elle
    amortit l'écart entre le rang 1 et le rang 2 pour ne pas laisser
    un seul index dominer totalement le classement final.

    Le matching entre les deux listes de résultats se fait par
    `chunk_id` (unique par tranche de document, voir retrieval.py) :
    texte et doc_id seraient suffisants ici, mais chunk_id évite tout
    risque de confusion si jamais deux tranches de deux documents
    différents avaient un texte identique."""

    def __init__(self, lexical_index, semantic_index, k_rrf: int = 60, fetch_k: int = 20):
        self.lexical_index = lexical_index
        self.semantic_index = semantic_index
        self.k_rrf = k_rrf
        self.fetch_k = fetch_k  # combien de résultats on va chercher dans CHAQUE sous-index

    def search(self, query: str, top_k: int = 3) -> list:
        lexical_results = self.lexical_index.search(query, top_k=self.fetch_k)
        semantic_results = self.semantic_index.search(query, top_k=self.fetch_k)

        rrf_scores = {}
        by_chunk_id = {}
        n_sources = 0
        for results in (lexical_results, semantic_results):
            if not results:
                continue
            n_sources += 1
            for rank, r in enumerate(results, start=1):
                cid = r["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k_rrf + rank)
                by_chunk_id[cid] = r

        # Score brut RRF : petit et sur une échelle qui n'a rien à voir
        # avec un score BM25 ou un cosinus (max théorique = 1/(k_rrf+1)
        # PAR index qui classe le passage n°1, donc ~0.016 avec 2 index
        # et k_rrf=60). On le renormalise par ce max théorique pour
        # retomber sur une échelle [0, 1] cohérente avec `min_score`
        # côté agent/UI (qui reste calibré pensant "0.12 = seuil
        # raisonnable", comme pour BM25Index/TfidfIndex).
        max_possible = n_sources / (self.k_rrf + 1) if n_sources else 1.0

        order = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            {**by_chunk_id[cid], "score": score / max_possible}
            for cid, score in order
        ]

"""
retrieval.py
------------
Recherche documentaire simple (TF-IDF + similarité cosinus), 100% NumPy,
pour ajouter un mode "question-réponse à partir de documents" par-dessus
le générateur LSTM.

Deux façons de répondre (voir app.py, endpoint /api/ask) :

  - EXTRACTIF (par défaut, `mode="extractive"`) : on renvoie tel quel le
    passage le plus proche de la question. Fiable — le texte renvoyé
    vient réellement du document, il n'est pas halluciné.

  - GÉNÉRATIF (`mode="generative"`) : on injecte les passages trouvés
    dans le prompt et on laisse le LSTM continuer le texte. À prendre
    avec des pincettes : ce LSTM est un petit modèle de langage entraîné
    uniquement à prédire le mot suivant sur du texte brut, SANS aucun
    exemple de question-réponse. Rien ne garantit qu'il s'appuie
    correctement sur le contexte fourni plutôt que d'inventer — c'est un
    ajout expérimental, pas un système de QA fiable.

TF-IDF : chaque passage (et chaque question) est représenté par un
vecteur où chaque dimension correspond à un mot du vocabulaire, pondéré
par sa fréquence dans le passage (TF) et son caractère plus ou moins
rare dans l'ensemble des passages (IDF — un mot présent partout, comme
"le" ou "de", est peu informatif et reçoit un poids faible). La
similarité entre la question et un passage est simplement le cosinus
entre leurs deux vecteurs.
"""

import os
import re
from collections import Counter

import numpy as np

# Mots vides français : très fréquents, peu informatifs pour juger de la
# pertinence d'un passage. Sans ce filtre, une question totalement hors
# sujet peut quand même récolter un score non nul juste en partageant
# "le", "la", "de"... avec un passage — ce qui donnerait une fausse
# impression de pertinence.
_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "au",
    "aux", "en", "dans", "sur", "pour", "par", "avec", "sans", "ce",
    "cette", "ces", "cet", "il", "elle", "ils", "elles", "je", "tu",
    "nous", "vous", "on", "qui", "que", "quoi", "où", "est", "sont",
    "être", "avoir", "a", "ont", "se", "son", "sa", "ses", "leur",
    "leurs", "ne", "pas", "plus", "ou", "mais", "donc", "or", "ni",
    "car", "comme", "si", "quel", "quelle", "quels", "quelles",
}


def _tokenize(text: str) -> list:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def load_documents_from_dir(directory: str) -> list:
    """Charge tous les .txt d'un dossier : un fichier = un document,
    découpé en passages (paragraphes séparés par une ligne vide)."""
    passages = []
    if not os.path.isdir(directory):
        return passages
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(directory, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                passages.append({"doc_id": fname, "text": para})
    return passages


class TfidfIndex:
    """Index TF-IDF construit une fois pour toutes sur une liste de
    passages ; `search()` peut ensuite être appelé pour chaque question."""

    def __init__(self, passages: list):
        self.passages = passages
        self.vocab = {}
        self.idf = None
        self.doc_vectors = None
        self._build()

    def _build(self):
        tokenized = [_tokenize(p["text"]) for p in self.passages]
        n_docs = len(tokenized)

        df = Counter()
        for toks in tokenized:
            for w in set(toks):
                df[w] += 1

        self.vocab = {w: i for i, w in enumerate(df.keys())}
        # IDF lissé (+1 partout) pour éviter les divisions par zéro et les
        # poids infinis sur un mot présent dans un seul passage.
        self.idf = np.array([
            np.log((1 + n_docs) / (1 + df[w])) + 1.0 for w in self.vocab
        ])

        vecs = np.zeros((max(n_docs, 1), len(self.vocab)))
        for row, toks in enumerate(tokenized):
            counts = Counter(toks)
            for w, c in counts.items():
                j = self.vocab.get(w)
                if j is not None:
                    vecs[row, j] = c * self.idf[j]

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = vecs / norms

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        counts = Counter(_tokenize(text))
        for w, c in counts.items():
            j = self.vocab.get(w)
            if j is not None:
                vec[j] = c * self.idf[j]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(self, query: str, top_k: int = 3) -> list:
        """Retourne jusqu'à `top_k` passages, triés par similarité
        décroissante, en ignorant les scores nuls (aucun mot en commun)."""
        if not self.passages:
            return []
        qvec = self._vectorize(query)
        scores = self.doc_vectors @ qvec
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"doc_id": self.passages[i]["doc_id"], "text": self.passages[i]["text"],
             "score": float(scores[i])}
            for i in order if scores[i] > 1e-9
        ]

"""
retrieval.py
------------
Recherche documentaire pour /api/ask. Deux évolutions par rapport à la
V1 (TF-IDF + découpage par paragraphe) :

  1. CHUNKING PAR TRANCHES AVEC CHEVAUCHEMENT (`chunk_text`) au lieu
     d'un découpage par paragraphe (`\\n\\s*\\n`). Le découpage par
     paragraphe a un défaut : un paragraphe peut faire 3 mots ou 3
     pages selon la façon dont le fichier source a été écrit — ça ne
     dit rien sur la quantité d'information qu'il contient. On préfère
     ici des tranches de taille RÉGULIÈRE (par défaut 200 tokens), afin
     que le score BM25 (qui est sensible à la longueur du document,
     voir plus bas) reste comparable d'un passage à l'autre. Le
     chevauchement (50 tokens par défaut) évite qu'une réponse qui
     tombait pile à la frontière entre deux tranches se retrouve coupée
     en deux morceaux chacun trop faible pour remonter dans le top-k.

  2. BM25 (`BM25Index`) au lieu du TF-IDF brut (`TfidfIndex`, conservé
     ci-dessous pour comparaison/référence, plus utilisé par défaut).
     Deux différences concrètes avec le TF-IDF :
       - SATURATION DE FRÉQUENCE : en TF-IDF, un mot qui apparaît 10
         fois dans un passage compte 10x plus qu'un mot qui apparaît 1
         fois. En BM25, ce poids SATURE (paramètre k1) : passer de 1 à
         2 occurrences compte beaucoup, de 9 à 10 presque plus rien —
         un mot répété artificiellement ne peut plus "gonfler"
         indéfiniment le score d'un passage.
       - NORMALISATION PAR LA LONGUEUR (paramètre b) : un passage plus
         long a mécaniquement plus de chances de contenir les mots de
         la question, sans être forcément plus PERTINENT pour autant.
         BM25 corrige ce biais via la longueur relative à la longueur
         moyenne des passages (`avgdl`) — c'est justement ce qui rend
         le chunking à taille fixe utile : avgdl a un sens.

Les deux classes exposent la même interface `search(query, top_k) ->
list[dict(doc_id, text, score, chunk_id)]`, utilisée telle quelle par
`agent.SearchTool` : l'agent ReAct ne sait pas s'il interroge un
TF-IDF, un BM25 ou un index d'embeddings (voir embeddings.py) — c'est
tout l'intérêt de la séparation agent / outil mise en place dans
agent.py.
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


# --------------------------------------------------------------------
# Chunking par tranches avec chevauchement
# --------------------------------------------------------------------
def _word_tokenize_with_spans(text: str) -> list:
    """Tokenisation par mots qui garde de quoi reconstruire le texte
    original (espaces/ponctuation compris), contrairement à `_tokenize`
    (qui jette la ponctuation et les stopwords — parfait pour indexer,
    mais on perdrait le texte source si on l'utilisait pour découper)."""
    return re.findall(r"\S+", text)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50,
               encode_fn=None, decode_fn=None) -> list:
    """Découpe `text` en tranches de `chunk_size` tokens, avec un
    chevauchement de `overlap` tokens entre deux tranches consécutives.

    Par défaut, le "token" est un mot (regex `\\S+`) : ça ne demande
    aucune dépendance et donne une bonne approximation. Si on fournit
    `encode_fn`/`decode_fn` (typiquement `bpe.encode`/`bpe.decode`,
    voir `load_documents_from_dir`), le découpage se fait sur les
    VRAIS tokens BPE du modèle — plus fidèle, mais nécessite d'avoir
    déjà un vocabulaire BPE entraîné (`merges`/`stoi`/`itos`).

    `overlap` doit être strictement inférieur à `chunk_size`, sinon la
    fenêtre n'avance jamais (boucle infinie) : on lève une erreur
    explicite plutôt que de planter silencieusement.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) doit être < chunk_size ({chunk_size})")

    if encode_fn is not None:
        tokens = encode_fn(text)
        join = decode_fn
    else:
        tokens = _word_tokenize_with_spans(text)
        join = " ".join

    n = len(tokens)
    if n == 0:
        return []

    stride = chunk_size - overlap
    chunks = []
    start = 0
    while start < n:
        window = tokens[start:start + chunk_size]
        chunks.append(join(window).strip())
        if start + chunk_size >= n:
            break
        start += stride
    return [c for c in chunks if c]


def load_documents_from_dir(directory: str, chunk_size: int = 200, overlap: int = 50,
                             encode_fn=None, decode_fn=None) -> list:
    """Charge tous les .txt d'un dossier : un fichier = un document,
    découpé en tranches de `chunk_size` tokens avec chevauchement de
    `overlap` tokens (voir `chunk_text`) — plus de découpage par
    paragraphe. `chunk_id` numérote les tranches dans l'ORDRE du
    document, ce qui sert à faire correspondre les résultats entre
    plusieurs index différents (voir `HybridIndex` dans embeddings.py)."""
    passages = []
    if not os.path.isdir(directory):
        return passages
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(directory, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap,
                             encode_fn=encode_fn, decode_fn=decode_fn)
        for i, chunk in enumerate(chunks):
            passages.append({"doc_id": fname, "chunk_id": f"{fname}#{i}", "text": chunk})
    return passages


# --------------------------------------------------------------------
# BM25 — index par défaut
# --------------------------------------------------------------------
class BM25Index:
    """Index BM25 (Okapi BM25, variante '+1' pour un IDF toujours
    positif), 100% NumPy, vectorisé.

    k1 (défaut 1.5) contrôle la vitesse de saturation de la fréquence
    d'un terme (plus k1 est grand, plus une répétition compte
    longtemps avant de saturer). b (défaut 0.75) contrôle la force de
    la normalisation par la longueur du document (0 = désactivée,
    1 = normalisation complète). Ce sont les valeurs par défaut
    standard de la littérature (Robertson & Zaragoza, 2009)."""

    def __init__(self, passages: list, k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = None
        self.tf = None          # (n_docs, vocab_size) fréquences brutes
        self.doc_len = None     # (n_docs,) nombre de tokens indexables par passage
        self.avgdl = 0.0
        self._build()

    def _build(self):
        tokenized = [_tokenize(p["text"]) for p in self.passages]
        n_docs = len(tokenized)

        df = Counter()
        for toks in tokenized:
            for w in set(toks):
                df[w] += 1
        self.vocab = {w: i for i, w in enumerate(df.keys())}
        V = len(self.vocab)

        # IDF "BM25+1" : log(1 + (N - n(w) + 0.5) / (n(w) + 0.5)). Le
        # "+1" à l'intérieur du log évite un IDF négatif pour les mots
        # présents dans plus de la moitié des passages (possible avec
        # la formule BM25 classique sur un petit corpus) — un mot très
        # fréquent doit peser proche de 0, jamais pénaliser un passage.
        self.idf = np.zeros(V)
        for w, i in self.vocab.items():
            self.idf[i] = np.log(1 + (n_docs - df[w] + 0.5) / (df[w] + 0.5))

        self.tf = np.zeros((max(n_docs, 1), V))
        for row, toks in enumerate(tokenized):
            counts = Counter(toks)
            for w, c in counts.items():
                self.tf[row, self.vocab[w]] = c

        self.doc_len = self.tf.sum(axis=1)
        self.avgdl = float(self.doc_len.mean()) if n_docs else 0.0

    def search(self, query: str, top_k: int = 3) -> list:
        if not self.passages or self.avgdl == 0:
            return []
        q_terms = [w for w in _tokenize(query) if w in self.vocab]
        if not q_terms:
            return []
        idxs = [self.vocab[w] for w in q_terms]

        tf_q = self.tf[:, idxs]                                   # (n_docs, n_query_terms)
        idf_q = self.idf[idxs]                                    # (n_query_terms,)
        length_norm = (1 - self.b + self.b * self.doc_len / self.avgdl)  # (n_docs,)

        numerator = tf_q * (self.k1 + 1)
        denominator = tf_q + self.k1 * length_norm[:, None]
        # Un terme absent du passage (tf=0) donne numérateur=0, donc ne
        # contribue rien au score, quel que soit le dénominateur — pas
        # de division par zéro à gérer explicitement.
        term_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                                 where=denominator > 0)
        scores = (term_scores * idf_q[None, :]).sum(axis=1)       # (n_docs,)

        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"doc_id": self.passages[i]["doc_id"], "chunk_id": self.passages[i]["chunk_id"],
             "text": self.passages[i]["text"], "score": float(scores[i])}
            for i in order if scores[i] > 1e-9
        ]


# --------------------------------------------------------------------
# TF-IDF — conservé pour référence/comparaison, plus utilisé par
# défaut dans app.py (voir BM25Index ci-dessus).
# --------------------------------------------------------------------
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

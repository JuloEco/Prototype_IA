"""
data.py
-------
Fonctions génériques de préparation des données, indépendantes du choix
de tokenizer. La tokenisation elle-même (BPE) vit dans bpe.py — ce
fichier ne s'occupe que de charger le texte et de découper la séquence
encodée en fenêtres pour l'entraînement du LSTM.

Différence avec la version feedforward : avant, une fenêtre de
`block_size` tokens ne produisait qu'UNE seule cible (le token juste
après la fenêtre). Avec un RNN entraîné par BPTT, chaque pas de temps
de la fenêtre produit sa propre prédiction (le token suivant à CE
pas-là) — donc Y a maintenant la même forme que X : Y[i, t] est le
token qui suit X[i, t]. C'est ce qui rend le BPTT nettement plus
efficace en données : une fenêtre de longueur `block_size` fournit
`block_size` exemples d'entraînement au lieu d'un seul.
"""

import numpy as np


def load_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_dataset(encoded: np.ndarray, block_size: int):
    """Découpe la séquence encodée en fenêtres de `block_size` tokens
    (X) alignées avec leur cible décalée d'un cran (Y), peu importe si
    les tokens sont des caractères, des mots ou des morceaux BPE."""
    n = len(encoded) - block_size
    X = np.zeros((n, block_size), dtype=np.int64)
    Y = np.zeros((n, block_size), dtype=np.int64)
    for i in range(n):
        X[i] = encoded[i:i + block_size]
        Y[i] = encoded[i + 1:i + block_size + 1]
    return X, Y


def train_val_split(X, Y, val_fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

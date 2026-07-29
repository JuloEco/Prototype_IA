"""
bpe.py
------
Byte-Pair Encoding (BPE) — la technique de tokenisation utilisée par GPT
et la plupart des LLM modernes. C'est le compromis entre deux extrêmes :

  - niveau CARACTÈRE : vocabulaire minuscule (~40 tokens), mais chaque
    token porte très peu de sens, et les séquences sont longues.
  - niveau MOT : chaque token porte beaucoup de sens, mais le vocabulaire
    explose (autant de tokens que de mots distincts), et un petit corpus
    n'offre pas assez de répétitions pour bien apprendre chaque mot rare.

BPE construit un vocabulaire de "sous-mots" en partant des caractères et
en fusionnant itérativement la paire de tokens adjacents la PLUS
FRÉQUENTE dans le corpus, jusqu'à atteindre une taille de vocabulaire
cible. Résultat : les mots fréquents ("le", "était", "horloge"...)
finissent par former un seul token, tandis que les mots rares ou
inconnus se décomposent en plusieurs morceaux réutilisables (souvent
proches de préfixes/suffixes/racines) plutôt que de planter.

Étapes :
  1. Pré-tokenisation (`pretokenize`) : découpe le texte en unités
     "mot + espace précédent" à la façon de GPT-2 — l'espace fait partie
     du token suivant plutôt que d'être un caractère séparé. Ça élimine
     le besoin de règles de recollage à la détokénisation : il suffit de
     concaténer les morceaux décodés, l'espacement est déjà encodé dedans.
  2. Apprentissage des fusions (`train_bpe`) sur les pré-tokens.
  3. Encodage d'un nouveau texte (`encode`) en appliquant les fusions
     apprises, dans l'ordre où elles ont été apprises.
"""

import re
from collections import Counter

# Un pré-token = un saut de ligne isolé, OU un mot (optionnellement
# précédé d'une espace), OU un caractère de ponctuation (optionnellement
# précédé d'une espace). L'espace fait partie du token qui suit.
_PRETOKEN_PATTERN = re.compile(r"\n| ?\w+| ?[^\w\s]", re.UNICODE)

UNK_TOKEN = "<unk>"


def pretokenize(text: str) -> list:
    return _PRETOKEN_PATTERN.findall(text)


def _get_pair_counts(splits: dict, word_freqs: Counter) -> Counter:
    pair_counts = Counter()
    for word, freq in word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts


def _merge_pair(pair, splits: dict, word_freqs: Counter):
    a, b = pair
    merged = a + b
    for word in word_freqs:
        symbols = splits[word]
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        splits[word] = new_symbols


def train_bpe(pretokens: list, vocab_size: int, verbose: bool = False):
    """Apprend les fusions BPE à partir d'une liste de pré-tokens.

    Retourne :
      - merges : liste ordonnée de paires fusionnées (l'ORDRE compte :
        c'est cet ordre qui sera réutilisé pour encoder du texte inédit)
      - vocab  : liste ordonnée de tous les tokens du vocabulaire final
        (caractères de base, puis chaque fusion apprise, dans l'ordre)
    """
    # Le saut de ligne reste un token spécial, jamais fusionné avec autre
    # chose (ça n'aurait pas de sens de fusionner "\n" avec un mot).
    word_freqs = Counter(t for t in pretokens if t != "\n")

    splits = {word: list(word) for word in word_freqs}
    base_chars = sorted(set(ch for word in word_freqs for ch in word))

    vocab = list(base_chars) + ["\n", UNK_TOKEN]
    merges = []

    target_merges = max(0, vocab_size - len(vocab))

    for step in range(target_merges):
        pair_counts = _get_pair_counts(splits, word_freqs)
        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            break  # plus aucune paire répétée : pas de fusion utile à faire

        _merge_pair(best_pair, splits, word_freqs)
        merges.append(best_pair)
        vocab.append(best_pair[0] + best_pair[1])

        if verbose and step % 20 == 0:
            print(f"  fusion {step:>4}: {best_pair} -> "
                  f"'{best_pair[0] + best_pair[1]}' (vue {best_count} fois)")

    return merges, vocab


def encode_word(word: str, merge_ranks: dict) -> list:
    """Applique les fusions apprises à UN pré-token, dans l'ordre de
    priorité (les fusions apprises le plus tôt s'appliquent en premier —
    c'est l'algorithme standard d'encodage BPE, identique à celui de
    GPT-2)."""
    if word == "\n":
        return ["\n"]

    symbols = list(word)
    while len(symbols) >= 2:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        ranked = [(merge_ranks[p], p) for p in pairs if p in merge_ranks]
        if not ranked:
            break
        _, best_pair = min(ranked, key=lambda x: x[0])

        a, b = best_pair
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(a + b)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    return symbols


def encode(text: str, merges: list, stoi: dict) -> list:
    """Encode un texte complet en indices de tokens BPE."""
    merge_ranks = {pair: i for i, pair in enumerate(merges)}
    ids = []
    for word in pretokenize(text):
        for piece in encode_word(word, merge_ranks):
            ids.append(stoi.get(piece, stoi[UNK_TOKEN]))
    return ids


def decode(ids: list, itos: dict) -> str:
    """Décode une liste d'indices en texte. Contrairement aux versions
    précédentes, il n'y a AUCUNE règle de recollage à appliquer : les
    espaces sont déjà encodés dans les tokens eux-mêmes (pré-tokenisation
    à la GPT-2), donc une simple concaténation suffit."""
    return "".join(itos.get(int(i), "") for i in ids)

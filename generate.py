"""
generate.py
-----------
Génère du texte token par token (morceaux BPE), avec les mêmes leviers
que la version précédente : température, top-k, top-p.

Changement clé par rapport à la version feedforward : il n'y a PLUS de
fenêtre de contexte fixe à reconstituer/padder à chaque pas. Le LSTM
fait circuler un état (h, c) : on "joue" le prompt dans le modèle une
fois, token par token, pour obtenir l'état après le prompt, puis on
continue à générer en mettant à jour cet état à chaque nouveau token —
exactement comme le ferait un humain qui lit une phrase en la découvrant
mot par mot, sans jamais avoir besoin de revenir en arrière ni de
"remplir" un début de phrase avec du vide.
"""

import argparse
import os
import numpy as np

from model import LSTMLM
from bpe import encode, decode, UNK_TOKEN

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.npz")


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return probs
    logits = np.log(probs + 1e-12) / max(temperature, 1e-4)
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _apply_top_k(probs: np.ndarray, top_k: int) -> np.ndarray:
    if not top_k or top_k <= 0 or top_k >= len(probs):
        return probs
    kth_value = np.partition(probs, -top_k)[-top_k]
    filtered = np.where(probs >= kth_value, probs, 0.0)
    return filtered / filtered.sum()


def _apply_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    if not top_p or top_p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, top_p) + 1
    cutoff = min(cutoff, len(probs))
    keep_indices = order[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[keep_indices] = probs[keep_indices]
    return filtered / filtered.sum()


def _apply_repetition_penalty(probs: np.ndarray, generated_ids: list,
                               penalty: float) -> np.ndarray:
    """Style CTRL : chaque token déjà généré (prompt inclus) voit sa
    probabilité divisée par `penalty` avant renormalisation. Ça n'aide
    pas la GRAMMAIRE (ce n'est pas son rôle), mais ça évite qu'un petit
    LSTM reste "collé" sur un token ou une courte boucle à haute
    probabilité — un travers classique des petits modèles de langage."""
    if not penalty or penalty == 1.0 or not generated_ids:
        return probs
    probs = probs.copy()
    for tid in set(generated_ids):
        probs[tid] /= penalty
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


def _block_repeated_ngrams(probs: np.ndarray, generated_ids: list, n: int) -> np.ndarray:
    """Interdit tout token qui recréerait un n-gramme déjà vu dans la
    séquence générée jusqu'ici (technique standard "no-repeat-ngram",
    utilisée par ex. par Hugging Face `generate`). Complète la pénalité
    de répétition : celle-ci atténue, ceci interdit purement et
    simplement les boucles de N tokens ou plus."""
    if not n or n <= 0 or len(generated_ids) < n - 1:
        return probs
    prefix = tuple(generated_ids[-(n - 1):])
    banned = set()
    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i + n - 1]) == prefix:
            banned.add(generated_ids[i + n - 1])
    if not banned:
        return probs
    filtered = probs.copy()
    for tid in banned:
        filtered[tid] = 0.0
    total = filtered.sum()
    if total <= 0:
        # Bloquer TOUS les candidats mènerait à une impasse : mieux vaut
        # laisser passer une répétition que planter la génération.
        return probs
    return filtered / total


def _apply_context_bias(probs: np.ndarray, context_ids: set, strength: float) -> np.ndarray:
    """Multiplie par `strength` la probabilité de tout token présent dans
    `context_ids` (le passage documentaire injecté dans le prompt), puis
    renormalise. Ce n'est PAS un vrai mécanisme de copie (comme dans un
    modèle pointer-generator) : ça ne force rien, ça incline juste la
    balance vers le vocabulaire du contexte plutôt que vers le style
    "par défaut" appris sur le corpus d'entraînement. Sans ça, un LSTM
    entraîné sur un corpus au style différent (littéraire) du contexte
    (scientifique, ici) dérive presque toujours vers SON style à lui au
    bout de quelques tokens, plutôt que de rester sur le sujet fourni."""
    if not context_ids or not strength or strength == 1.0:
        return probs
    probs = probs.copy()
    idx = np.array(list(context_ids))
    probs[idx] *= strength
    total = probs.sum()
    if total > 0:
        probs /= total
    return probs


def generate(model: LSTMLM, merges: list, stoi: dict, itos: dict, prompt: str = "",
             length: int = 60, temperature: float = 0.8,
             top_k: int = 0, top_p: float = 1.0,
             repetition_penalty: float = 1.3, no_repeat_ngram_size: int = 3,
             context_bias_text: str = None, context_bias_strength: float = 2.5,
             seed: int = None) -> str:
    """`length` est un nombre de tokens BPE à générer après le prompt.

    `repetition_penalty` (>1.0 = actif) et `no_repeat_ngram_size` (>0 =
    actif) sont deux garde-fous appliqués AVANT température/top-k/top-p :
    ils ne rendent pas le modèle plus intelligent, mais évitent les
    boucles/répétitions dégénérées auxquelles un petit LSTM est sujet,
    surtout à basse température.

    `context_bias_text` (optionnel) : un texte (typiquement un passage
    documentaire retrouvé par retrieval.py) dont le vocabulaire est
    favorisé pendant TOUTE la génération — voir `_apply_context_bias`.
    Sert au mode génératif du chat documentaire (app.py /api/ask) pour
    limiter la dérive hors-sujet ; n'a pas d'effet en génération libre
    (context_bias_text=None par défaut)."""
    rng = np.random.default_rng(seed)

    prompt_ids = list(encode(prompt, merges, stoi)) if prompt else []
    if not prompt_ids:
        # Il faut au moins un token pour amorcer l'état du LSTM.
        pad_token = "\n" if "\n" in stoi else next(iter(stoi))
        prompt_ids = [stoi[pad_token]]

    context_ids = set()
    if context_bias_text:
        context_ids = {i for i in encode(context_bias_text, merges, stoi) if i in itos}

    h, c = model.init_state(batch_size=1)
    probs = None
    for tok_id in prompt_ids:
        probs, h, c = model.step(np.array([tok_id]), h, c)

    output_ids = list(prompt_ids)
    for _ in range(length):
        p = probs[0]
        p = _apply_repetition_penalty(p, output_ids, repetition_penalty)
        p = _block_repeated_ngrams(p, output_ids, no_repeat_ngram_size)
        p = _apply_context_bias(p, context_ids, context_bias_strength)
        p = _apply_temperature(p, temperature)
        p = _apply_top_k(p, top_k)
        p = _apply_top_p(p, top_p)

        next_id = int(rng.choice(len(p), p=p))
        output_ids.append(next_id)
        probs, h, c = model.step(np.array([next_id]), h, c)

    return decode(output_ids, itos)


def main():
    parser = argparse.ArgumentParser(description="Génère du texte avec le modèle LSTM entraîné (BPE).")
    parser.add_argument("--prompt", type=str, default="Il était")
    parser.add_argument("--length", type=int, default=80, help="Nombre de tokens BPE à générer")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 = désactivé")
    parser.add_argument("--top-p", type=float, default=1.0, help="1.0 = désactivé")
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                         help="1.0 = désactivé, >1.0 pénalise les tokens déjà générés")
    parser.add_argument("--no-repeat-ngram", type=int, default=3,
                         help="0 = désactivé ; interdit de répéter un n-gramme déjà vu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Aucun modèle trouvé à {MODEL_PATH}. Lance d'abord `python train.py`.")
        return

    model, stoi, itos, merges = LSTMLM.load(MODEL_PATH)
    text = generate(model, merges, stoi, itos, prompt=args.prompt, length=args.length,
                     temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                     repetition_penalty=args.repetition_penalty,
                     no_repeat_ngram_size=args.no_repeat_ngram,
                     seed=args.seed)
    print(text)


if __name__ == "__main__":
    main()

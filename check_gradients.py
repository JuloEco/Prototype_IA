"""
check_gradients.py
-------------------
Vérifie que backward() (le BPTT écrit à la main) calcule les vrais
gradients de la loss, par comparaison avec une estimation par
différences finies. Deux passes :

  1. Sans dropout (dropout_rate=0) : valide les équations du LSTM et le
     BPTT de base (portes, cellule mémoire, sortie).
  2. Avec dropout activé (training=True) : valide que le gradient
     traverse correctement le masque de dropout à CHAQUE pas de temps.
     Pour que la comparaison soit valide, on fixe le masque à une valeur
     identique pour tous les appels forward (via `dropout_seed`).

Lancement :
    python check_gradients.py
"""

import numpy as np
from model import LSTMLM

np.random.seed(0)
FIXED_DROPOUT_SEED = 123  # mêmes masques (tous pas de temps confondus) à chaque appel forward


import re

import numpy as np
from model import LSTMLM

np.random.seed(0)
FIXED_DROPOUT_SEED = 123  # mêmes masques (toutes couches et pas de temps confondus) à chaque appel forward


def _get_param(model, name):
    """Les poids des portes vivent dans des listes (model.Wg[l],
    model.bg[l]) plutôt que des attributs directs, à cause de
    l'empilement de couches — ce helper fait le pont entre le nom "plat"
    utilisé par `parameters()`/`backward()` (ex: "Wg0", "bg1") et l'objet
    réel à modifier pour la différence finie."""
    m = re.fullmatch(r"(Wg|bg)(\d+)", name)
    if m:
        attr, idx = m.group(1), int(m.group(2))
        return getattr(model, attr)[idx]
    return getattr(model, name)


def numerical_gradient(model, param_name, X, Y, training, eps=1e-5):
    param = _get_param(model, param_name)
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])

    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        param[idx] = original_value + eps
        probs_plus, _ = model.forward(X, training=training, dropout_seed=FIXED_DROPOUT_SEED)
        loss_plus = model.loss(probs_plus, Y)

        param[idx] = original_value - eps
        probs_minus, _ = model.forward(X, training=training, dropout_seed=FIXED_DROPOUT_SEED)
        loss_minus = model.loss(probs_minus, Y)

        param[idx] = original_value
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        it.iternext()

    return grad


def relative_error(a, b):
    # Plancher plus élevé qu'un simple epsilon anti-division-par-zéro :
    # avec le BPTT, certains gradients sont légitimement minuscules
    # (~1e-8, portes quasi saturées). Sur ces entrées, le "signal" est
    # dominé par le bruit de l'approximation par différences finies
    # elle-même (pas par une vraie erreur de dérivation) — l'erreur
    # relative brute y exploserait pour rien. On la plafonne donc par un
    # plancher au dénominateur, pratique standard du gradient checking.
    return np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-6)


def run_check(label: str, dropout_rate: float, training: bool, num_layers: int = 2):
    vocab_size, block_size, embed_dim, hidden_dim = 6, 3, 4, 8
    batch_size = 5

    # dtype=float64 EXPLICITEMENT ICI : le modèle utilise float32 par
    # défaut pour la vitesse d'entraînement (voir model.py), mais la
    # différence finie (eps=1e-5) perdrait toute précision utile dans le
    # bruit d'arrondi float32 — ce test doit rester en double précision
    # quel que soit le dtype par défaut du modèle.
    model = LSTMLM(vocab_size, block_size, embed_dim, hidden_dim,
                    num_layers=num_layers, dropout_rate=dropout_rate, seed=1,
                    dtype=np.float64)
    X = np.random.randint(0, vocab_size, size=(batch_size, block_size))
    Y = np.random.randint(0, vocab_size, size=(batch_size, block_size))

    dseed = FIXED_DROPOUT_SEED if training else None
    probs, cache = model.forward(X, training=training, dropout_seed=dseed)
    analytical_grads = model.backward(cache, Y)

    print(f"\n=== {label} (num_layers={num_layers}) ===")
    print(f"{'Paramètre':<8} {'Erreur relative max':<22} {'Statut'}")
    print("-" * 45)

    all_ok = True
    param_names = ["C", "W2", "b2"] + [f"{p}{l}" for l in range(num_layers) for p in ("Wg", "bg")]
    for name in param_names:
        numerical = numerical_gradient(model, name, X, Y, training)
        analytical = analytical_grads[name]
        err = relative_error(analytical, numerical)
        max_err = err.max()
        status = "OK" if max_err < 1e-4 else "❌ SUSPECT"
        if max_err >= 1e-4:
            all_ok = False
        print(f"{name:<8} {max_err:<22.2e} {status}")

    return all_ok


def main():
    ok1 = run_check("Sans dropout (équations LSTM + BPTT de base)", dropout_rate=0.0, training=False, num_layers=1)
    ok2 = run_check("Avec dropout activé (masques fixés pour le test)", dropout_rate=0.3, training=True, num_layers=1)
    ok3 = run_check("Empilé, sans dropout (BPTT inter-couches)", dropout_rate=0.0, training=False, num_layers=2)
    ok4 = run_check("Empilé, avec dropout (masques fixés)", dropout_rate=0.3, training=True, num_layers=3)

    print()
    if ok1 and ok2 and ok3 and ok4:
        print("✅ Tous les gradients sont corrects (1, 2 et 3 couches, avec et sans dropout).")
    else:
        print("❌ Au moins un gradient est incorrect — ne pas entraîner tant que ce n'est pas corrigé.")


if __name__ == "__main__":
    main()

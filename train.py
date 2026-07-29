"""
train.py
--------
Entraîne le modèle de langage — VERSION LSTM :
  - architecture RNN à portes (LSTM), rétropropagée dans le temps
    (BPTT), à la place de l'ancien réseau feedforward à fenêtre fixe
  - tokenisation BPE, learning rate schedule, dropout, early stopping :
    inchangés dans leur principe

Changement notable par rapport à la version feedforward : le hack de
padding du corpus (préfixer avec des tokens "\\n" pour habituer le
modèle aux contextes courts) A DISPARU. Il n'a plus lieu d'être : le
LSTM ne dépend pas d'une fenêtre de contexte figée à remplir, et chaque
fenêtre d'entraînement démarre déjà d'un état caché nul (h=0, c=0) —
exactement comme un prompt de génération, aussi court soit-il. C'est le
LSTM lui-même qui résout le problème que le padding rafistolait.

Lancement :
    python train.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from data import load_corpus, make_dataset, train_val_split
from bpe import pretokenize, train_bpe, encode, decode
from model import LSTMLM
from generate import generate
from retrieval import load_documents_from_dir

BPE_VOCAB_SIZE = 600      # ↑ de 300 à 600 : moins de mots "reconstruits" à partir de
                          # fragments rares mal appris (cause fréquente des mots inventés)
BLOCK_SIZE = 32           # ↑ de 24 à 32 : un peu plus de contexte rétropropagé d'un coup
EMBED_DIM = 48            # ↑ de 32 : plus de place pour encoder chaque sous-mot
HIDDEN_DIM = 256          # ↑ de 192 : plus de capacité pour la grammaire/l'accord
NUM_LAYERS = 2            # LSTM empilé — voir model.py pour la justification.
DTYPE = np.float32        # ↓ de float64 (défaut NumPy implicite) : ~1.9x plus rapide sur les
                          # matmuls du LSTM, sans perte de qualité observable pour ce genre de
                          # modèle (Adam et le clipping de gradient sont peu sensibles à cette
                          # précision). Voir model.py pour le piège à éviter (mélange de dtypes).
DROPOUT_RATE = 0.25       # dropout entre les couches ET avant la sortie
WEIGHT_DECAY = 1e-4       # régularisation AdamW découplée sur Wg{l}/W2 (voir model.py)

# BATCH_SIZE et N_STEPS : ↑ le batch, ↓ les pas dans les MÊMES proportions
# (384 = 6x, 2000 = 12000/6) pour voir EXACTEMENT le même nombre total
# d'exemples d'entraînement (768 000) qu'avant, donc "apprendre autant" —
# mais avec 6x moins de boucles Python par pas de temps sur toute la durée
# de l'entraînement, dont le coût fixe (indépendant de la taille du batch)
# domine largement le temps de calcul pur des matmuls à cette échelle.
# Gain mesuré (avec float32 ci-dessus) : ~2.5x plus rapide au total. Si ta
# machine a plusieurs cœurs CPU, le gain peut être encore plus net : les
# matmuls d'un batch plus grand se parallélisent bien avec BLAS multi-thread
# (OpenBLAS, utilisé par défaut par les roues NumPy), alors que le coût de
# la boucle Python, lui, ne profite d'aucun parallélisme.
BATCH_SIZE = 384          # ↑ de 64
N_STEPS = 2000            # ↓ de 12000, dans la même proportion que BATCH_SIZE
BASE_LR = 1.4e-3          # ↑ de 8e-4 : un batch plus grand donne un gradient moins bruité,
                          # ce qui tolère (et bénéficie d'ordinaire d'a) un LR plus élevé —
                          # règle empirique de "linear scaling", appliquée ici prudemment
                          # (x1.75 plutôt que x6) pour rester dans une zone déjà validée par
                          # le clipping de gradient existant.
WARMUP_STEPS = 100        # ↓ de 300, proportionnellement à N_STEPS
MIN_LR_RATIO = 0.1
EVAL_EVERY = 50           # ↓ de 200, proportionnellement à N_STEPS (même fréquence relative)

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.npz")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "models", "training_curve.png")
LR_PLOT_PATH = os.path.join(os.path.dirname(__file__), "models", "lr_schedule.png")


def get_lr(step, total_steps, base_lr, warmup_steps=WARMUP_STEPS, min_lr_ratio=MIN_LR_RATIO):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def get_batch(X, Y, batch_size, rng):
    idx = rng.integers(0, len(X), size=batch_size)
    return X[idx], Y[idx]


def eval_loss(model, X_val, Y_val, batch_size=256):
    """Perte de validation calculée par mini-lots (need_cache=False) :
    la mémoire utilisée ne dépend plus de la taille de X_val, seulement
    de `batch_size` — important dès que le corpus grandit."""
    total_loss = 0.0
    total_n = 0
    for start in range(0, len(X_val), batch_size):
        Xb = X_val[start:start + batch_size]
        Yb = Y_val[start:start + batch_size]
        probs, _ = model.forward(Xb, training=False, need_cache=False)
        total_loss += model.loss(probs, Yb) * len(Xb)
        total_n += len(Xb)
    return total_loss / max(1, total_n)


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    rng = np.random.default_rng(0)

    text = load_corpus(CORPUS_PATH)

    # Le vocabulaire BPE est appris sur corpus.txt + data/documents/, pas
    # sur corpus.txt seul : sinon, tout caractère absent de corpus.txt
    # (symboles scientifiques comme ₂, →, ⁺... dans les documents utilisés
    # par le chat) n'a AUCUNE représentation dans le vocabulaire — pas
    # même au niveau caractère — et devient <unk> dès qu'il apparaît dans
    # le contexte injecté en mode génératif (voir app.py /api/ask).
    # Le LSTM, lui, continue à s'entraîner UNIQUEMENT sur corpus.txt (ligne
    # plus bas) : élargir le vocabulaire ne change pas son style d'écriture,
    # ça évite juste de casser silencieusement l'encodage d'un texte externe.
    doc_passages = load_documents_from_dir(DOCS_DIR)
    docs_text = "\n\n".join(p["text"] for p in doc_passages)
    vocab_training_text = text + "\n\n" + docs_text if docs_text else text
    pretoks = pretokenize(vocab_training_text)

    print(f"Apprentissage du vocabulaire BPE (cible : {BPE_VOCAB_SIZE} tokens)...")
    if docs_text:
        print(f"  (vocabulaire élargi avec {len(doc_passages)} passages de {DOCS_DIR})")
    merges, vocab = train_bpe(pretoks, vocab_size=BPE_VOCAB_SIZE)
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for i, tok in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulaire BPE : {vocab_size} tokens ({len(merges)} fusions apprises)")

    token_ids = encode(text, merges, stoi)
    encoded = np.array(token_ids, dtype=np.int64)
    print(f"Corpus : {len(text)} caractères -> {len(encoded)} tokens BPE "
          f"({len(text) / len(encoded):.2f} caractères/token)")

    X, Y = make_dataset(encoded, BLOCK_SIZE)
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y, val_fraction=0.1)
    print(f"Fenêtres d'entraînement : {len(X_train)}  |  validation : {len(X_val)}  "
          f"(chacune fournit {BLOCK_SIZE} cibles, une par pas de temps)")

    model = LSTMLM(vocab_size, BLOCK_SIZE, EMBED_DIM, HIDDEN_DIM,
                    num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE, seed=42, dtype=DTYPE)

    history = {"step": [], "train_loss": [], "val_loss": [], "lr": []}

    # Early stopping : identique dans le principe à la version précédente
    # — on garde une copie des poids au moment où la perte de validation
    # est la plus basse jamais observée, et on restaure CETTE version à
    # la fin, peu importe combien de pas suivent.
    best_val_loss = float("inf")
    best_params = None
    best_step = 0

    print(f"\nEntraînement sur {N_STEPS} pas "
          f"(dropout={DROPOUT_RATE}, LR de base={BASE_LR}, warmup={WARMUP_STEPS})...\n")

    for step in range(1, N_STEPS + 1):
        lr = get_lr(step, N_STEPS, BASE_LR)

        Xb, Yb = get_batch(X_train, Y_train, BATCH_SIZE, rng)
        probs, cache = model.forward(Xb, training=True)
        train_loss = model.loss(probs, Yb)
        grads = model.backward(cache, Yb)
        model.adam_step(grads, lr=lr, weight_decay=WEIGHT_DECAY)

        if step % EVAL_EVERY == 0 or step == 1:
            val_loss = eval_loss(model, X_val, Y_val)
            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_params = {k: v.copy() for k, v in model.parameters().items()}
                marker = " <- meilleur jusqu'ici"
            else:
                marker = ""

            sample = generate(model, merges, stoi, itos, prompt="Il était", length=25,
                               temperature=0.8, top_k=20, top_p=0.9)
            sample_display = sample.replace("\n", " / ")
            print(f"[{step:>5}/{N_STEPS}] lr={lr:.2e}  "
                  f"train_loss={train_loss:.3f}  val_loss={val_loss:.3f}{marker}")
            print(f"           échantillon : \"{sample_display}\"")

    if best_params is not None:
        print(f"\nRestauration du meilleur modèle : pas {best_step} "
              f"(val_loss={best_val_loss:.3f}), plutôt que le dernier pas ({N_STEPS}).")
        model.set_parameters(best_params)

    model.save(MODEL_PATH, BLOCK_SIZE, stoi, itos, merges)
    print(f"Modèle sauvegardé : {MODEL_PATH}")

    plt.figure(figsize=(9, 5))
    plt.plot(history["step"], history["train_loss"], label="Perte (entraînement)", color="#10a37f")
    plt.plot(history["step"], history["val_loss"], label="Perte (validation)", color="#e5534b")
    plt.xlabel("Pas d'entraînement")
    plt.ylabel("Cross-entropy (nats)")
    plt.title("Courbe d'apprentissage (LSTM + BPE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Courbe de perte sauvegardée : {PLOT_PATH}")

    plt.figure(figsize=(9, 3.5))
    plt.plot(history["step"], history["lr"], color="#7dd3c0")
    plt.xlabel("Pas d'entraînement")
    plt.ylabel("Learning rate")
    plt.title("Learning rate schedule (warmup + décroissance cosinus)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(LR_PLOT_PATH, dpi=120)
    print(f"Courbe du LR sauvegardée : {LR_PLOT_PATH}")


if __name__ == "__main__":
    main()

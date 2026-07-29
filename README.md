# ✍️ Générateur de texte — 100% NumPy, sans framework (v4 — LSTM)

Un modèle de langage entraîné par descente de gradient, avec une
rétropropagation **écrite entièrement à la main**. Cette version
remplace le réseau **feedforward à fenêtre fixe** des versions
précédentes par un **LSTM** (Long Short-Term Memory), rétropropagé dans
le temps (BPTT — BackProp Through Time), toujours à la main.

## 🕰️ Petit historique

| Version | Tokenisation | Architecture | Ce qu'on a observé |
|---|---|---|---|
| v1 | caractère par caractère | feedforward | Vocabulaire minuscule (~40), peu de sens par token |
| v2 | mot par mot | feedforward | Plus de sens par token, mais vocabulaire qui explose |
| v3 | BPE (sous-mots) | feedforward, fenêtre fixe | Compromis vocabulaire compact / sens par token, mais toujours limité par une fenêtre de contexte rigide et un plafond de capacité |
| **v4** | **BPE (sous-mots)** | **LSTM (RNN à portes)** | Le contexte n'est plus une fenêtre figée mais un état qui circule token par token — plus de piège du padding, et une architecture nettement plus expressive |

## 🧠 Pourquoi un LSTM ?

Le réseau feedforward de la v3 regardait toujours EXACTEMENT
`block_size` tokens de contexte, tous injectés d'un coup dans une seule
couche cachée. Deux limites en découlaient :

- **Une fenêtre rigide.** Un prompt plus court que `block_size` devait
  être "complété" artificiellement (padding), ce qui créait des
  contextes jamais vus pendant l'entraînement.
- **Un plafond de capacité.** Une seule couche cachée sans notion
  explicite d'ordre séquentiel a du mal à apprendre des dépendances qui
  s'étendent sur plusieurs mots — augmenter `hidden_dim` ou le dropout
  n'y changeait rien, c'était un plafond architectural.

Un **RNN** traite la séquence token par token et fait circuler un état
d'un pas de temps au suivant, ce qui règle le premier problème. Un LSTM
va plus loin qu'un RNN "vanille" : il ajoute une **cellule mémoire**
(`c`) et trois **portes** (entrée, oubli, sortie) qui contrôlent
explicitement ce qui entre dans la mémoire, ce qu'on oublie, et ce qu'on
expose en sortie. C'est ce mécanisme de portes qui permet au gradient
de circuler sur des séquences plus longues sans s'effondrer (*vanishing
gradient*) — le problème classique qui limite les RNN sans portes.

### Les équations (`model.py`)

À chaque pas de temps `t`, avec `x_t` l'embedding du token courant et
`z_t = [x_t, h_{t-1}]` :

```
i_t = sigmoid(z_t Wi + bi)        # porte d'entrée : qu'est-ce qu'on laisse rentrer ?
f_t = sigmoid(z_t Wf + bf)        # porte d'oubli   : qu'est-ce qu'on garde de la mémoire ?
o_t = sigmoid(z_t Wo + bo)        # porte de sortie : qu'est-ce qu'on expose ?
g_t = tanh(z_t Wg + bg)           # candidat mémoire
c_t = f_t * c_{t-1} + i_t * g_t   # nouvelle cellule mémoire
h_t = o_t * tanh(c_t)             # nouvel état caché (nourrit la prédiction)
```

Les quatre matrices sont concaténées en une seule (`Wg`, forme
`(embed_dim + hidden_dim, 4*hidden_dim)`) pour n'avoir qu'un seul
produit matriciel par pas de temps — pure optimisation, les équations
restent identiques.

**Astuce d'initialisation** : le biais de la porte d'oubli est
initialisé à +1 (au lieu de 0). Ça pousse le réseau, au tout début de
l'entraînement, vers "je garde ma mémoire" plutôt que "j'oublie tout" —
ça évite de tuer le gradient dès les premiers pas.

### La rétropropagation : BPTT

`backward()` rejoue la séquence à l'envers, du dernier pas de temps au
premier. Deux gradients circulent d'un pas à l'autre : `dh_next` (à
travers `h_{t-1}`) et `dc_next` (à travers `c_{t-1}`). Un même jeu de
poids reçoit une contribution de gradient à CHAQUE pas de temps, et ces
contributions s'additionnent — c'est la seule vraie différence
structurelle avec la rétropropagation d'un feedforward classique.

`check_gradients.py` vérifie cette dérivation par comparaison avec des
différences finies (avec et sans dropout) :

```bash
python check_gradients.py
```

## ⚡ Accélérer l'entraînement sans rien perdre à l'apprentissage

Trois changements, mesurés (pas juste théoriques) sur ce projet, sans
toucher à la quantité de données vues ni à la capacité du modèle :

1. **`DTYPE = np.float32`** (au lieu du float64 par défaut de NumPy) :
   ~1.9x plus rapide sur les matmuls du LSTM, mesuré. Le piège classique
   avec un changement pareil : mélanger du float32 et du float64 dans
   une même opération fait que NumPy PROMEUT silencieusement tout en
   float64, annulant le gain sans erreur ni avertissement. `model.py`
   utilise donc `self.dtype` explicitement pour CHAQUE allocation (états
   initiaux, masques de dropout, accumulateurs de gradient, état
   d'Adam...), pas seulement pour les poids.

2. **`BATCH_SIZE` plus grand, `N_STEPS` réduit dans la même proportion**
   (384 et 2000 au lieu de 64 et 12000 — même nombre total d'exemples
   vus : 768 000 dans les deux cas). La boucle Python sur les pas de
   temps a un coût FIXE par appel à `forward`/`backward`, indépendant de
   la taille du batch — en traitant plus d'exemples par appel, ce coût
   fixe est amorti sur davantage de données. Gain mesuré : encore ~1.3x,
   au-delà de ce qu'apporte déjà le float32 seul.

3. **Moins d'appels à coût fixe dans la boucle chaude** : les embeddings
   de tout le batch sont maintenant récupérés en un seul `self.C[X]`
   avant la boucle sur les pas de temps (plutôt qu'un `self.C[X[:, t]]`
   répété à chaque pas), et le gradient de l'embedding est accumulé sur
   toute la séquence puis appliqué en un seul `np.add.at` à la fin du
   BPTT (plutôt qu'un appel par pas de temps).

**Gain total mesuré sur ce projet : ~2.5x** (une configuration qui
prenait ~95 min en tourne ~38 min avec les nouveaux réglages). Si ta
machine a plusieurs cœurs CPU, le gain peut être encore plus net : les
matmuls d'un batch plus grand se parallélisent bien avec un BLAS
multi-thread (OpenBLAS, utilisé par défaut par les roues NumPy
installées via pip), alors que le coût de la boucle Python, lui, ne
profite d'aucun parallélisme quel que soit le nombre de cœurs.

Piste que j'ai testée et ÉCARTÉE après mesure : remplacer la
concaténation `[entrée, h_{t-1}]` + un seul matmul par deux matmuls
séparés (`entrée @ Wx + h_prev @ Wh`). Sur le papier, ça semble plus
direct — en pratique, mesuré, c'est légèrement PLUS LENT (deux appels
matmul au lieu d'un). Un rappel utile : sur ce genre d'optimisation, il
vaut mieux mesurer que supposer.

## 🧱 LSTM empilé (`num_layers`)

Le LSTM peut maintenant compter plusieurs couches (`NUM_LAYERS` dans
`train.py`, 2 par défaut). Une seule couche apprend une seule
transformation récurrente ; en empiler plusieurs donne au réseau la
possibilité d'apprendre des régularités de plus en plus abstraites
(une couche proche de la surface — accords, terminaisons — puis une
couche plus globale — cohérence sur la phrase), un peu comme empiler
des couches convolutives en vision apprend des motifs de plus en plus
abstraits.

Techniquement, chaque couche a ses propres poids de portes (`model.Wg`
et `model.bg` sont maintenant des LISTES, une entrée par couche), et le
BPTT (`backward()`) doit faire circuler deux flux de gradient distincts
à chaque pas de temps : un qui remonte le TEMPS (comme avant, d'un pas
à l'autre) et un qui descend les COUCHES (de la sortie vers
l'embedding, au même pas de temps). `check_gradients.py` vérifie cette
dérivation avec 1, 2 ET 3 couches.

Le dropout, déjà présent avant la projection finale, s'applique
maintenant aussi ENTRE les couches — plus utile qu'avec une seule
couche : empiler sans rien réguler entre elles favorise une couche à
apprendre les particularités exactes de l'autre plutôt que des
régularités générales.

## 🎉 Le piège du padding : résolu par l'architecture, pas par un rafistolage

Dans la v3, un prompt court comme "Il était" devait être complété
(padding) pour remplir la fenêtre fixe de `block_size` tokens, sous
peine de tomber sur un contexte jamais vu à l'entraînement. La v3
corrigeait ça en préfixant le corpus d'entraînement avec du padding —
un rafistolage qui marchait, mais qui traitait le symptôme.

Avec le LSTM, ce problème n'existe plus du tout, structurellement :

- **À l'entraînement**, chaque fenêtre de `block_size` tokens démarre
  d'un état caché nul (`h=0, c=0`) — le modèle apprend donc, dès le
  premier pas de temps de chaque fenêtre, à produire une prédiction
  sensée à partir d'un état "qui vient de commencer".
- **À la génération** (`generate.py`), on ne reconstitue plus aucune
  fenêtre : le prompt est joué token par token dans le modèle à partir
  du même état nul, et l'état circule ensuite librement, aussi longue
  que soit la génération. Un prompt de 2 tokens n'est donc plus un cas
  "hors distribution" à corriger — c'est simplement une séquence qui
  vient de commencer, exactement comme à l'entraînement.

Résultat : `data.py` et `train.py` n'ont plus aucune trace de logique de
padding, et `generate.py` n'a plus besoin de connaître `block_size` du
tout côté génération.

## 🛡️ Nouveau : le clipping de gradient

Le BPTT multiplie des gradients sur plusieurs pas de temps. Même avec
les portes du LSTM (qui limitent le *vanishing gradient*), un gradient
occasionnellement trop grand (*exploding gradient*) peut faire diverger
l'entraînement d'un coup — un risque que le feedforward de la v3
n'avait pas. `model.py` ajoute donc un **clipping de la norme globale du
gradient** (plafonnée à 5.0 par défaut) dans `adam_step`, une précaution
standard pour l'entraînement de RNN/LSTM.

## 🔬 Entraînement : une cible par pas de temps, pas juste une par fenêtre

Autre changement notable (`data.py`) : dans la v3, une fenêtre de
`block_size` tokens ne produisait qu'UNE seule cible (le token juste
après la fenêtre). Avec le LSTM entraîné par BPTT, chaque pas de temps
de la fenêtre produit sa propre prédiction — `Y` a donc maintenant la
même forme que `X` (`Y[i, t]` est le token qui suit `X[i, t]`). Une
fenêtre de longueur `block_size` fournit ainsi `block_size` exemples
d'entraînement au lieu d'un seul, ce qui rend chaque pas de descente de
gradient nettement plus riche en signal.

## 📁 Structure

```
text-generator-nn/
├── bpe.py                  # Algorithme BPE : pré-tokenisation, apprentissage, encode/decode (inchangé)
├── data.py                   # Fenêtres d'entraînement (X, Y séquentiels) — plus de padding
├── model.py                    # LSTM : forward pas-à-pas, BPTT manuel, Adam + clipping
├── check_gradients.py            # Vérification numérique du BPTT (avec et sans dropout)
├── train.py                        # Entraînement BPE + LSTM + LR schedule + courbes
├── generate.py                       # Génération incrémentale (état h, c) + température/top-k/top-p
├── app.py                              # Interface web (chat documentaire + génération libre)
├── retrieval.py                          # Recherche TF-IDF sur data/documents/ (voir section Chat)
├── data/corpus.txt                       # Texte d'entraînement du LSTM (remplace-le par le tien !)
├── data/documents/                       # Fichiers .txt sur lesquels "discuter" (voir section Chat)
├── models/                                 # Modèle + vocabulaire BPE, générés par train.py
├── templates/ , static/                      # Interface web
└── requirements.txt
```

## 💬 Chat documentaire (`retrieval.py`, `/api/ask`)

L'onglet **Chat** de l'interface web (`python app.py`) répond à des
questions à partir des fichiers `.txt` placés dans `data/documents/` —
un document par fichier, chacun découpé en paragraphes. Aucun entraînement
supplémentaire n'est nécessaire : la recherche se fait par **TF-IDF +
similarité cosinus**, 100% NumPy, recalculée au démarrage de `app.py`.

Deux modes, choisis dans les réglages avancés du chat :

- **Extractif (par défaut)** : la réponse est le passage le plus proche
  de la question, recopié tel quel. Fiable — impossible d'halluciner
  puisque rien n'est généré — mais littéral : pas de reformulation.
- **Génératif** : les passages trouvés sont injectés dans le prompt du
  LSTM, qui continue le texte pour formuler une réponse. À prendre avec
  des pincettes : ce LSTM est entraîné à prédire le mot suivant sur
  `data/corpus.txt`, **jamais sur des paires question/réponse** — rien
  ne garantit qu'il s'appuie sur le contexte fourni plutôt que
  d'inventer. C'est d'autant plus vrai si `data/corpus.txt` et
  `data/documents/` n'ont pas le même style ou sujet (ex. : LSTM entraîné
  sur un texte littéraire, questions sur un document scientifique) : le
  modèle sait rarement rester "sur le sujet" du contexte fourni.

Si aucune question ne dépasse le seuil de confiance (score TF-IDF trop
faible — la question est probablement hors sujet), le chat répond
explicitement qu'il n'a rien trouvé plutôt que d'improviser.

**Limite à garder en tête** : ce n'est pas un vrai système de
Retrieval-Augmented Generation comme on l'entend habituellement (pas
d'embeddings sémantiques, pas de LLM pour reformuler) — la recherche
TF-IDF ne capte que des mots partagés, pas le sens. Une question qui
reformule totalement une idée sans reprendre son vocabulaire risque de
ne rien trouver même si la réponse est présente dans les documents.


## ▶️ Utilisation

```bash
pip install -r requirements.txt

python check_gradients.py    # optionnel
python train.py               # entraîne le vocabulaire BPE + le LSTM
python generate.py --prompt "Il était" --length 60 --temperature 0.8 --top-k 20 --top-p 0.9
python app.py                   # interface web, http://127.0.0.1:5002 — onglets Chat + Génération libre
```

## 🎛️ Réglages (`train.py`)

| Paramètre | Rôle |
|---|---|
| `BPE_VOCAB_SIZE` | Taille cible du vocabulaire de sous-mots (300 par défaut) |
| `BLOCK_SIZE` | Longueur de troncature du BPTT — nombre de pas de temps rétropropagés d'un coup (24) |
| `EMBED_DIM` / `HIDDEN_DIM` | Taille des embeddings / de l'état caché du LSTM |
| `NUM_LAYERS` | Nombre de couches de LSTM empilées (2 par défaut — voir section dédiée plus haut) |
| `DTYPE` | Précision des calculs (`np.float32` par défaut, voir section "Accélérer l'entraînement") |
| `DROPOUT_RATE` | Fraction de l'état caché "éteinte" à l'entraînement, tirée à chaque pas de temps |
| `BASE_LR` | Taux d'apprentissage de base (8e-4) |
| `N_STEPS` | Nombre de pas d'entraînement — l'early stopping garde le meilleur point automatiquement |

Notez que `BLOCK_SIZE` n'a plus le même sens que dans la v3 : ce n'est
plus une fenêtre de contexte qui limite ce que le modèle peut "voir" à
la génération (le LSTM n'a aucune limite de ce type, son état peut
circuler indéfiniment), c'est uniquement la longueur sur laquelle le
gradient est rétropropagé à chaque pas d'entraînement — un compromis
mémoire/vitesse classique du BPTT tronqué, sans lien avec la capacité du
modèle à générer sur de longues séquences.

## ✏️ Utiliser TON propre texte

Remplace `data/corpus.txt` par un texte plus long (vise 20 000+ mots)
pour de meilleurs résultats — le LSTM lève une partie du plafond de
capacité rencontré avec le feedforward de la v3, mais reste un modèle
simple : plus de données aide toujours autant qu'avant.

## 🚀 Pour aller plus loin

- **Empiler ENCORE plus de couches** ou augmenter `hidden_dim` : la
  capacité continue de monter, mais l'entraînement ralentit et le
  risque de surapprentissage grandit d'autant sur un petit corpus.
- **Auto-attention / Transformer** : l'étape suivante logique après le
  LSTM pour capturer des dépendances encore plus longues sans le goulot
  d'étranglement d'un état caché de taille fixe.
- **Vocabulaire BPE plus grand** (500-1000 tokens) avec un corpus plus
  gros : meilleur compromis compression/expressivité.
- **Weight decay** : en complément du dropout et du clipping de
  gradient, pour compléter la régularisation.

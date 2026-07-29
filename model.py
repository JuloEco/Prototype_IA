"""
model.py
--------
Le modèle de langage — VERSION LSTM EMPILÉ (plusieurs couches).

Pourquoi un LSTM (rappel) ?
---------------------------
Le modèle feedforward des versions précédentes regardait toujours
EXACTEMENT `block_size` tokens de contexte, tous injectés d'un coup
dans une seule couche cachée — pas de notion d'ordre au-delà de la
position dans la fenêtre, et une fenêtre totalement rigide. Un LSTM
traite la séquence token par token et fait circuler un état (h, c)
d'un pas de temps au suivant, avec trois portes (entrée, oubli,
sortie) qui contrôlent explicitement ce qui entre dans la mémoire, ce
qu'on oublie, et ce qu'on expose en sortie — ce mécanisme de portes
permet au gradient de circuler sur de longues séquences sans
s'effondrer (vanishing gradient).

Pourquoi EMPILER plusieurs couches (`num_layers > 1`) ?
---------------------------------------------------------
Une seule couche de LSTM apprend UNE transformation récurrente : "vu ce
que je sais déjà (h) et le token courant, comment mettre à jour mon
état ?". Empiler des couches donne au réseau la possibilité d'apprendre
des transformations de plus en plus abstraites — la couche 1 peut
capter des motifs proches de la surface (accords, terminaisons), la
couche 2 des régularités plus globales (cohérence sur toute une
phrase). C'est le levier de CAPACITÉ le plus direct pour ce projet.

Équations d'une cellule (identiques par couche) — avec `entrée_t` =
embedding du token si couche 0, sinon sortie (après dropout) de la
couche précédente au même pas de temps t :

    z_t = [entrée_t, h_{t-1}]
    i_t = sigmoid(z_t Wi + bi)        porte d'entrée
    f_t = sigmoid(z_t Wf + bf)        porte d'oubli
    o_t = sigmoid(z_t Wo + bo)        porte de sortie
    g_t = tanh(z_t Wg + bg)           candidat mémoire
    c_t = f_t * c_{t-1} + i_t * g_t   nouvelle cellule
    h_t = o_t * tanh(c_t)             nouvel état caché

BPTT multi-couches (`backward`) : pour chaque pas de temps t (de la fin
vers le début), on parcourt les couches de la DERNIÈRE vers la
PREMIÈRE. Deux flux de gradient à ne pas confondre :
  - dh_next[l] / dc_next[l] : gradient qui revient du FUTUR pour la
    couche l (à travers h_{t-1}, c_{t-1} de cette même couche).
  - `incoming` : gradient qui descend des couches SUPÉRIEURES au même
    pas de temps t (à travers l'entrée de la couche l+1, qui est
    justement la sortie — avec dropout — de la couche l).
`check_gradients.py` vérifie cette dérivation par différences finies
(toujours en float64, voir plus bas), avec num_layers > 1.

Pourquoi `dtype=np.float32` par défaut (nouveau) ?
----------------------------------------------------
NumPy calcule par défaut en float64 (double précision) dès qu'on
n'impose rien — deux fois plus d'octets à déplacer en mémoire pour
chaque multiplication de matrices que le strict nécessaire pour
entraîner un LSTM de cette taille. Sur CPU, un forward+backward mesuré
en float32 tourne significativement plus vite qu'en float64 pour un
LSTM de cette taille (mesuré : ~1.8x sur les matmuls), sans perte de
qualité d'apprentissage observable : le clipping de gradient et Adam
(qui normalise déjà par des variances estimées) sont peu sensibles à
cette précision. Le PIÈGE à éviter : mélanger du float32 et du
float64 dans une même opération fait que NumPy PROMEUT silencieusement
tout en float64 — annulant le gain sans erreur ni avertissement. C'est
pour ça que toutes les allocations (états initiaux, masques de dropout,
accumulateurs de gradient, état d'Adam...) utilisent explicitement
`self.dtype` ci-dessous, pas seulement les poids.

Pour la VÉRIFICATION des gradients (`check_gradients.py`), le float32
est en revanche un mauvais choix : la différence finie (eps=1e-5)
perdrait toute précision utile dans le bruit d'arrondi float32. Le
modèle y est donc explicitement instancié en `dtype=np.float64`.
"""

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LSTMLM:
    def __init__(self, vocab_size: int, block_size: int,
                 embed_dim: int = 32, hidden_dim: int = 192,
                 num_layers: int = 2, dropout_rate: float = 0.25, seed: int = 42,
                 dtype=np.float32):
        self.vocab_size = vocab_size
        self.block_size = block_size          # longueur de troncature du BPTT à l'entraînement
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dtype = np.dtype(dtype)

        rng = np.random.default_rng(seed)
        H = hidden_dim

        self.C = rng.normal(0, 0.08, size=(vocab_size, embed_dim)).astype(self.dtype)

        # Une matrice de portes par couche (ordre des 4 portes empilées :
        # [i, f, o, g]). La couche 0 reçoit l'embedding (dim embed_dim) +
        # son propre h_{t-1} (dim H) ; les couches suivantes reçoivent la
        # sortie (avec dropout) de la couche précédente (dim H) + leur
        # propre h_{t-1} (dim H).
        self.Wg = []
        self.bg = []
        for l in range(num_layers):
            in_dim = embed_dim if l == 0 else H
            gate_in = in_dim + H
            Wl = rng.normal(0, (2.0 / gate_in) ** 0.5, size=(gate_in, 4 * H)).astype(self.dtype)
            bl = np.zeros(4 * H, dtype=self.dtype)
            bl[H:2 * H] = 1.0  # biais de la porte d'oubli initialisé à 1 (voir docstring du module)
            self.Wg.append(Wl)
            self.bg.append(bl)

        self.W2 = rng.normal(0, (2.0 / H) ** 0.5, size=(H, vocab_size)).astype(self.dtype)
        self.b2 = np.zeros(vocab_size, dtype=self.dtype)

        self._adam_state = {
            name: {"m": np.zeros_like(p), "v": np.zeros_like(p), "t": 0}
            for name, p in self.parameters().items()
        }
        self._dropout_rng = np.random.default_rng(seed + 1)

    # ------------------------------------------------------------------
    def parameters(self):
        params = {"C": self.C, "W2": self.W2, "b2": self.b2}
        for l in range(self.num_layers):
            params[f"Wg{l}"] = self.Wg[l]
            params[f"bg{l}"] = self.bg[l]
        return params

    def set_parameters(self, params: dict):
        """Réassigne tous les poids depuis un dict au format de
        `parameters()` — utilisé par l'early stopping dans train.py pour
        restaurer la meilleure copie des poids (Wg{l}/bg{l} vivent dans
        des listes, donc un simple `setattr` ne suffit pas)."""
        self.C = params["C"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
        for l in range(self.num_layers):
            self.Wg[l] = params[f"Wg{l}"]
            self.bg[l] = params[f"bg{l}"]

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int):
        """État initial (h_0, c_0) de CHAQUE couche — des vecteurs nuls,
        comme un LSTM qui n'a encore rien vu. Retourne deux LISTES de
        longueur num_layers (une entrée par couche)."""
        H = self.hidden_dim
        h = [np.zeros((batch_size, H), dtype=self.dtype) for _ in range(self.num_layers)]
        c = [np.zeros((batch_size, H), dtype=self.dtype) for _ in range(self.num_layers)]
        return h, c

    # ------------------------------------------------------------------
    def step(self, x_ids: np.ndarray, h_prev: list, c_prev: list):
        """Un seul pas de temps, SANS dropout (génération — voir
        generate.py). x_ids : (N,) indices de tokens. h_prev/c_prev :
        listes (une par couche), comme retourné par init_state/step."""
        H = self.hidden_dim
        x = self.C[x_ids]
        new_h, new_c = [], []
        entree = x
        for l in range(self.num_layers):
            z = np.concatenate([entree, h_prev[l]], axis=1)
            a = z @ self.Wg[l] + self.bg[l]
            ai, af, ao, ag = a[:, :H], a[:, H:2 * H], a[:, 2 * H:3 * H], a[:, 3 * H:4 * H]
            i, f, o, g = _sigmoid(ai), _sigmoid(af), _sigmoid(ao), np.tanh(ag)
            c = f * c_prev[l] + i * g
            h = o * np.tanh(c)
            new_h.append(h)
            new_c.append(c)
            entree = h  # pas de dropout à l'inférence : alimente la couche suivante telle quelle

        logits = new_h[-1] @ self.W2 + self.b2
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs, new_h, new_c

    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray, training: bool = False, dropout_seed=None,
                need_cache: bool = True):
        """X : (N, T) indices de tokens. Retourne probs_seq (N,T,V) et un
        cache pour backward.

        `need_cache=False` : à utiliser quand on n'appellera JAMAIS
        `backward()` sur ce passage (typiquement l'évaluation sur un gros
        jeu de validation) — évite de stocker, pour chaque pas de temps
        ET chaque couche, une dizaine de tableaux (N, H)."""
        N, T = X.shape
        H = self.hidden_dim
        L = self.num_layers

        # Un seul "gather" de tous les embeddings du batch, plutôt qu'un
        # gather répété à chaque pas de temps (self.C[X[:, t]] dans une
        # boucle) : X est de toute façon connu en entier à l'avance, donc
        # rien n'empêche de le faire en une fois.
        emb_all = self.C[X]  # (N, T, E)

        if training and self.dropout_rate > 0:
            rng = np.random.default_rng(dropout_seed) if dropout_seed is not None else self._dropout_rng
            keep_prob = 1.0 - self.dropout_rate
            # Un masque indépendant par COUCHE et par PAS DE TEMPS.
            mask_all = (rng.random((L, T, N, H)) < keep_prob).astype(self.dtype) / self.dtype.type(keep_prob)
        else:
            mask_all = None

        h = [np.zeros((N, H), dtype=self.dtype) for _ in range(L)]
        c = [np.zeros((N, H), dtype=self.dtype) for _ in range(L)]
        probs_seq = np.zeros((N, T, self.vocab_size), dtype=self.dtype)
        steps = [[] for _ in range(L)] if need_cache else None   # steps[l][t] = cache couche l, pas t

        for t in range(T):
            entree = emb_all[:, t, :]
            for l in range(L):
                h_prev, c_prev = h[l], c[l]
                z = np.concatenate([entree, h_prev], axis=1)
                a = z @ self.Wg[l] + self.bg[l]
                ai, af, ao, ag = a[:, :H], a[:, H:2 * H], a[:, 2 * H:3 * H], a[:, 3 * H:4 * H]
                i_g, f_g, o_g, g_g = _sigmoid(ai), _sigmoid(af), _sigmoid(ao), np.tanh(ag)
                c_new = f_g * c_prev + i_g * g_g
                tanh_c = np.tanh(c_new)
                h_new = o_g * tanh_c

                if mask_all is not None:
                    mask_t = mask_all[l, t]
                    h_drop = h_new * mask_t
                else:
                    mask_t = None
                    h_drop = h_new

                if need_cache:
                    steps[l].append({
                        "h_prev": h_prev, "c_prev": c_prev, "z": z,
                        "i": i_g, "f": f_g, "o": o_g, "g": g_g,
                        "tanh_c": tanh_c,
                        "mask": mask_t if mask_t is not None else np.ones((N, H), dtype=self.dtype),
                        "h_drop": h_drop,
                    })

                h[l], c[l] = h_new, c_new
                entree = h_drop   # alimente la couche suivante (ou la sortie si l == L-1)

            logits = entree @ self.W2 + self.b2
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / exp.sum(axis=1, keepdims=True)
            probs_seq[:, t, :] = probs

            if need_cache:
                steps[L - 1][t]["probs"] = probs
                steps[L - 1][t]["top_h_drop"] = entree   # = h_drop de la dernière couche, entrée de W2

        cache = {"X": X, "steps": steps} if need_cache else None
        return probs_seq, cache

    # ------------------------------------------------------------------
    @staticmethod
    def loss(probs_seq: np.ndarray, Y: np.ndarray) -> float:
        """Cross-entropy moyennée sur TOUS les pas de temps ET tout le
        batch (Y a la même forme (N, T) que X : Y[:, t] est le token qui
        suit X[:, t]). Calculée en float64 quel que soit le dtype du
        modèle : c'est juste un scalaire affiché/loggué, la précision ne
        coûte rien ici et évite un affichage bruité en float32."""
        N, T, V = probs_seq.shape
        flat_probs = probs_seq.reshape(N * T, V).astype(np.float64)
        flat_Y = Y.reshape(N * T)
        correct = flat_probs[np.arange(N * T), flat_Y]
        return float(-np.log(correct + 1e-12).mean())

    # ------------------------------------------------------------------
    def backward(self, cache: dict, Y: np.ndarray) -> dict:
        """BPTT multi-couches (voir docstring du module pour le détail
        des deux flux de gradient : `dh_next`/`dc_next` à travers le
        temps, `incoming` à travers les couches)."""
        X = cache["X"]
        steps = cache["steps"]
        N, T = X.shape
        H = self.hidden_dim
        L = self.num_layers
        NT = N * T

        dC = np.zeros_like(self.C)
        dWg = [np.zeros_like(w) for w in self.Wg]
        dbg = [np.zeros_like(b) for b in self.bg]
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        # Gradient de l'embedding accumulé pour TOUTE la séquence d'un
        # coup (un seul np.add.at à la fin) plutôt qu'un np.add.at par
        # pas de temps — ce dernier a un coût fixe non négligeable
        # répété `block_size` fois par pas d'entraînement.
        dEmb_all = np.zeros((N, T, self.embed_dim), dtype=self.dtype)

        dh_next = [np.zeros((N, H), dtype=self.dtype) for _ in range(L)]
        dc_next = [np.zeros((N, H), dtype=self.dtype) for _ in range(L)]

        for t in reversed(range(T)):
            top_cache = steps[L - 1][t]
            probs = top_cache["probs"]

            dlogits = probs.astype(self.dtype).copy()
            dlogits[np.arange(N), Y[:, t]] -= 1.0
            dlogits /= NT

            dW2 += top_cache["top_h_drop"].T @ dlogits
            db2 += dlogits.sum(axis=0)

            incoming = dlogits @ self.W2.T   # gradient sur h_drop de la couche L-1 à ce pas t

            for l in reversed(range(L)):
                st = steps[l][t]

                dh = incoming * st["mask"] + dh_next[l]          # <- gradient du futur s'ajoute ici
                do = dh * st["tanh_c"]
                dc = dh * st["o"] * (1.0 - st["tanh_c"] ** 2) + dc_next[l]   # <- idem

                df = dc * st["c_prev"]
                di = dc * st["g"]
                dg = dc * st["i"]
                dc_prev = dc * st["f"]                            # <- devient dc_next[l] pour t-1

                dai = di * st["i"] * (1.0 - st["i"])
                daf = df * st["f"] * (1.0 - st["f"])
                dao = do * st["o"] * (1.0 - st["o"])
                dag = dg * (1.0 - st["g"] ** 2)

                da = np.concatenate([dai, daf, dao, dag], axis=1)

                dWg[l] += st["z"].T @ da
                dbg[l] += da.sum(axis=0)

                dz = da @ self.Wg[l].T
                in_dim = self.embed_dim if l == 0 else H
                dentree = dz[:, :in_dim]      # descend vers la couche l-1 (ou dC si l==0)
                dh_prev = dz[:, in_dim:]      # devient dh_next[l] pour le pas t-1

                dh_next[l] = dh_prev
                dc_next[l] = dc_prev

                if l == 0:
                    dEmb_all[:, t, :] = dentree
                else:
                    incoming = dentree         # devient le "incoming" de la couche l-1, MÊME pas t

        np.add.at(dC, X, dEmb_all)

        grads = {"C": dC, "W2": dW2, "b2": db2}
        for l in range(L):
            grads[f"Wg{l}"] = dWg[l]
            grads[f"bg{l}"] = dbg[l]
        return grads

    # ------------------------------------------------------------------
    def adam_step(self, grads: dict, lr: float = 3e-3,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                  clip_norm: float = 5.0, weight_decay: float = 0.0):
        """Adam + deux garde-fous utiles pour un RNN/LSTM (a fortiori
        empilé, où le BPTT multiplie des gradients sur plusieurs pas de
        temps ET plusieurs couches) :
          - clipping de la norme globale du gradient (évite l'exploding
            gradient qui ferait diverger l'entraînement d'un coup).
          - weight decay découplé (AdamW), appliqué UNIQUEMENT aux
            matrices de poids (Wg{l}, W2) — jamais aux biais ni à la
            table d'embedding C.

        Les accumulateurs `m`/`v` restent dans `self.dtype` (float32 par
        défaut) : Adam divise par sqrt(v) + eps, donc `eps` doit rester
        représentable sans s'écraser à zéro en float32 (1e-8 l'est très
        largement, float32 va jusqu'à ~1e-38)."""
        if clip_norm:
            total_sq = sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values())
            total_norm = total_sq ** 0.5
            if total_norm > clip_norm:
                scale = clip_norm / (total_norm + 1e-8)
                grads = {k: g * scale for k, g in grads.items()}

        decayed_params = {"W2"} | {f"Wg{l}" for l in range(self.num_layers)}

        params = self.parameters()
        for name, p in params.items():
            g = grads[name]
            state = self._adam_state[name]
            state["t"] += 1
            t = state["t"]

            state["m"] = beta1 * state["m"] + (1 - beta1) * g
            state["v"] = beta2 * state["v"] + (1 - beta2) * (g ** 2)

            m_hat = state["m"] / (1 - beta1 ** t)
            v_hat = state["v"] / (1 - beta2 ** t)

            if weight_decay and name in decayed_params:
                p -= lr * weight_decay * p

            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    def save(self, path: str, block_size: int, stoi: dict, itos: dict, merges: list):
        tokens = list(stoi.keys())
        merges_left = np.array([m[0] for m in merges], dtype=object)
        merges_right = np.array([m[1] for m in merges], dtype=object)

        arrays = dict(
            C=self.C, W2=self.W2, b2=self.b2,
            block_size=block_size,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            dtype_name=np.dtype(self.dtype).name,
            tokens=np.array(tokens, dtype=object),
            indices=np.array([stoi[t] for t in tokens]),
            merges_left=merges_left,
            merges_right=merges_right,
        )
        for l in range(self.num_layers):
            arrays[f"Wg{l}"] = self.Wg[l]
            arrays[f"bg{l}"] = self.bg[l]

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        num_layers = int(data["num_layers"]) if "num_layers" in data else 1
        dtype = np.dtype(str(data["dtype_name"])) if "dtype_name" in data else np.float64
        model = cls(
            vocab_size=int(data["vocab_size"]),
            block_size=int(data["block_size"]),
            embed_dim=int(data["embed_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            num_layers=num_layers,
            dropout_rate=float(data["dropout_rate"]) if "dropout_rate" in data else 0.0,
            dtype=dtype,
        )
        model.C = data["C"].astype(dtype)
        model.W2 = data["W2"].astype(dtype)
        model.b2 = data["b2"].astype(dtype)

        if "Wg0" in data:
            model.Wg = [data[f"Wg{l}"].astype(dtype) for l in range(num_layers)]
            model.bg = [data[f"bg{l}"].astype(dtype) for l in range(num_layers)]
        else:
            # Compatibilité avec un modèle sauvegardé par l'ancienne
            # version à une seule couche (clés "Wg"/"bg" sans suffixe).
            model.Wg = [data["Wg"].astype(dtype)]
            model.bg = [data["bg"].astype(dtype)]

        tokens = data["tokens"]
        indices = data["indices"]
        stoi = {str(t): int(i) for t, i in zip(tokens, indices)}
        itos = {int(i): str(t) for t, i in zip(tokens, indices)}

        merges = list(zip(
            [str(x) for x in data["merges_left"]],
            [str(x) for x in data["merges_right"]],
        ))

        return model, stoi, itos, merges

"""
agent.py
--------
Agent ReAct (Reasoning + Acting) pour /api/ask.

Objectif : ne PLUS injecter du texte dans un prompt de façon rigide
("à chaque question, on lance TF-IDF, point"), mais faire passer
l'agent par une boucle explicite et traçable :

    Thought  -> l'agent réfléchit à ce qu'il doit faire
    Action   -> il choisit un outil et l'appelle (ex: search["mot clé"])
    Observation -> il lit le résultat renvoyé par l'outil
    ... (la boucle peut recommencer : reformuler et rechercher à nouveau)
    Final Answer -> il formule la réponse finale

Chaque étape est enregistrée dans `trace` et renvoyée au client : c'est
ce qui distingue un agent d'un simple pipeline "retrieval -> template".

--------------------------------------------------------------------
Honnêteté sur une limite importante de CE projet précis
--------------------------------------------------------------------
Dans un agent ReAct "classique" (type LangChain), c'est le LLM
lui-même qui GÉNÈRE le texte "Thought: ... / Action: search[...]" à
chaque étape, et le framework se contente de parser cette sortie.
Ici, le modèle de langage est un petit LSTM entraîné uniquement à
prédire le mot suivant sur un corpus brut, SANS aucun exemple de
dialogue ni de format "Thought/Action" (voir generate.py) : lui
demander de générer un JSON ou un format structuré fiable n'a
quasiment aucune chance de marcher, et ce serait mentir sur les
capacités réelles du système que de faire semblant du contraire.

La "réflexion" (Thought -> Action) est donc ici une politique
explicite, écrite en Python et inspectable (`_decide`), plutôt que du
texte généré par le LSTM. Le LSTM n'intervient QUE là où le projet
l'utilisait déjà : dans le mode "generative" de la réponse finale.
Le point important, et ce qui répond à la demande, c'est que le
CONTRÔLEUR (Agent) est séparé de l'OUTIL (Tool) par une interface
claire : demain, `_decide` peut être remplacé par un appel à un vrai
LLM instruction-tuned sans toucher au reste de la boucle ni à
retrieval.py.
"""

from dataclasses import dataclass, field

from retrieval import TfidfIndex


# --------------------------------------------------------------------
# Outil : wrapper autour de TfidfIndex. L'agent ne connaît QUE cette
# interface (name, description, run) — il ignore tout de TF-IDF, des
# vecteurs, du cosinus, etc. On pourrait ajouter un second outil (une
# calculatrice, une recherche web...) sans changer une ligne de la
# boucle ReAct ci-dessous, juste en l'ajoutant à `self.tools`.
# --------------------------------------------------------------------
class SearchTool:
    name = "search"
    description = ("Cherche les passages les plus pertinents dans les documents "
                    "indexés (similarité TF-IDF). Action: search[\"mots clés\"]")

    def __init__(self, index: TfidfIndex):
        self.index = index

    def run(self, query: str, top_k: int = 3) -> list:
        if self.index is None:
            return []
        return self.index.search(query, top_k=top_k)


@dataclass
class ReActAgent:
    """Boucle ReAct bornée (au plus `max_steps` allers-retours
    Action -> Observation) au-dessus de SearchTool, avec formulation
    de la réponse finale déléguée à un `answerer` (fonction fournie par
    app.py, qui sait comment appeler le LSTM pour le mode génératif)."""

    tool: SearchTool
    answerer: callable  # (question, results, mode, **kwargs) -> (answer, note)
    min_score: float = 0.12
    max_steps: int = 2

    def run(self, question: str, memory=None, mode: str = "extractive",
            top_k: int = 3, **answerer_kwargs) -> dict:
        trace = []

        query = self._resolve_query(question, memory, trace)

        results, final_query = self._search_loop(query, top_k, trace)

        if not results:
            trace.append({
                "step": "final_answer",
                "content": "Aucun passage suffisamment pertinent trouvé.",
            })
            return {
                "answer": None,
                "sources": [],
                "note": ("Aucun passage suffisamment pertinent trouvé pour cette "
                         "question — mieux vaut ne rien répondre que d'improviser "
                         "à partir d'un document sans rapport."),
                "trace": trace,
            }

        trace.append({
            "step": "thought",
            "content": f"J'ai {len(results)} passage(s) pertinent(s) (meilleur score "
                       f"{results[0]['score']:.2f}). Je formule la réponse finale.",
        })
        answer, note = self.answerer(question, results, mode, **answerer_kwargs)
        trace.append({"step": "final_answer", "content": answer})

        return {"answer": answer, "sources": results, "note": note,
                "mode": mode, "trace": trace, "search_query": final_query}

    # ------------------------------------------------------------------
    # Thought : reformuler la question en requête de recherche, en
    # s'appuyant sur la mémoire pour résoudre les questions elliptiques
    # ("et pour Paris ?", "pourquoi ?") qui n'ont aucun sens hors
    # contexte pour un TF-IDF (qui ne voit que des sacs de mots).
    # ------------------------------------------------------------------
    _ELLIPTIC_MARKERS = {
        "ça", "cela", "ca", "il", "elle", "lui", "leur", "en", "y",
        "aussi", "et", "pourquoi", "comment", "sinon", "encore",
    }

    def _resolve_query(self, question: str, memory, trace: list) -> str:
        words = question.lower().split()
        # Elliptique = très court ET contient un marqueur de reprise
        # ("et ça ?", "pourquoi ?") — une question courte mais autonome
        # ("photosynthèse ?") n'a pas besoin d'être combinée au tour
        # précédent, la combiner ne ferait qu'ajouter du bruit.
        is_elliptic = len(words) <= 4 and any(w.strip("?!.,") in self._ELLIPTIC_MARKERS for w in words)
        if is_elliptic and memory is not None:
            previous = memory.last_user_question()
            if previous:
                trace.append({
                    "step": "thought",
                    "content": (f"La question '{question}' est courte/elliptique : "
                                 f"je la combine avec le tour précédent "
                                 f"('{previous}') pour la recherche."),
                })
                return f"{previous} {question}"
        trace.append({
            "step": "thought",
            "content": f"Je cherche des passages pertinents pour : '{question}'.",
        })
        return question

    # ------------------------------------------------------------------
    # Action / Observation, avec une reformulation si le premier essai
    # ne renvoie rien d'assez bon (boucle bornée par max_steps).
    # ------------------------------------------------------------------
    def _search_loop(self, query: str, top_k: int, trace: list):
        results = []
        for step_n in range(1, self.max_steps + 1):
            trace.append({"step": "action", "content": f'search["{query}"]'})
            raw_results = self.tool.run(query, top_k=top_k)
            good_results = [r for r in raw_results if r["score"] >= self.min_score]

            trace.append({
                "step": "observation",
                "content": (f"{len(raw_results)} passage(s) trouvé(s), "
                            f"{len(good_results)} au-dessus du seuil {self.min_score}."),
            })

            if good_results:
                return good_results, query

            if step_n < self.max_steps:
                new_query = self._broaden_query(query)
                if new_query == query:
                    break  # rien à changer, inutile de reboucler à l'identique
                trace.append({
                    "step": "thought",
                    "content": (f"Rien d'assez pertinent pour '{query}'. "
                                f"Je réessaie avec une requête élargie : '{new_query}'."),
                })
                query = new_query

        return results, query

    @staticmethod
    def _broaden_query(query: str) -> str:
        """Reformulation très simple : ne garder que les mots les plus
        longs (souvent les plus porteurs de sens), pour limiter le
        bruit d'un mot trop spécifique qui aurait fait chuter le score.
        Un vrai LLM ferait ici une reformulation sémantique ; un TF-IDF
        n'a que la forme des mots à se mettre sous la dent."""
        words = query.split()
        if len(words) <= 2:
            return query
        long_words = sorted(words, key=len, reverse=True)[: max(2, len(words) // 2)]
        return " ".join(long_words)

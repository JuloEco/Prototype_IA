"""
memory.py
---------
Mémoire conversationnelle explicite pour /api/ask.

Avant : chaque appel à /api/ask était traité isolément (une question =
une recherche TF-IDF = une réponse, sans savoir ce qui avait été dit
avant). Ce module ajoute une structure minimale mais réelle de gestion
de conversation :

  - Chaque message a un rôle (`user`, `assistant`, ou `system`) et un
    contenu, comme dans l'API Anthropic/OpenAI.
  - Chaque conversation est identifiée par un `conversation_id` (un
    UUID généré côté serveur au premier échange, renvoyé au client, qui
    le renvoie ensuite à chaque requête suivante).
  - L'historique est borné (`max_turns`) : on ne garde que les N
    derniers échanges, pour ne pas faire grossir indéfiniment le
    contexte injecté dans l'agent (important surtout en mode génératif,
    où tout ce contexte finit dans le prompt du LSTM).

Stockage : un simple dict en mémoire process (`_STORE`). C'est
volontairement basique — pas de base de données — puisque c'est un
prototype mono-process. Si l'app tourne un jour avec plusieurs workers
(gunicorn -w 4, par ex.), il faudra remplacer `_STORE` par un stockage
partagé (Redis, SQLite...) : la mémoire ne survivrait pas au routage
d'une requête vers un autre worker.
"""

import uuid
from datetime import datetime, timezone

VALID_ROLES = {"system", "user", "assistant"}

# conversation_id -> list[dict(role, content, timestamp)]
_STORE: dict[str, list[dict]] = {}


class ConversationMemory:
    """Vue sur l'historique d'UNE conversation. Ne stocke rien
    elle-même : lit/écrit dans le store global `_STORE`, ce qui permet
    de la recréer librement à chaque requête HTTP sans perdre l'état
    (Flask ne garde pas d'instance Python vivante entre deux requêtes)."""

    def __init__(self, conversation_id: str = None, max_turns: int = 6):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_turns = max_turns
        _STORE.setdefault(self.conversation_id, [])

    def add(self, role: str, content: str) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Rôle invalide : {role!r} (attendu : {VALID_ROLES})")
        _STORE[self.conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._trim()

    def _trim(self) -> None:
        """Ne garde que les derniers `max_turns` échanges user+assistant
        (les messages system, s'il y en a, sont toujours conservés en
        tête)."""
        history = _STORE[self.conversation_id]
        system_msgs = [m for m in history if m["role"] == "system"]
        turn_msgs = [m for m in history if m["role"] != "system"]
        # 1 "tour" = 1 message user + 1 message assistant -> 2*max_turns messages
        turn_msgs = turn_msgs[-2 * self.max_turns:]
        _STORE[self.conversation_id] = system_msgs + turn_msgs

    def history(self) -> list[dict]:
        """Copie de l'historique complet (avec timestamps), pour
        affichage/debug côté client par exemple."""
        return list(_STORE[self.conversation_id])

    def last_user_question(self) -> str | None:
        for m in reversed(_STORE[self.conversation_id]):
            if m["role"] == "user":
                return m["content"]
        return None

    def as_context_text(self, n_last: int = 4) -> str:
        """Rend les derniers échanges sous forme de texte brut
        `role: contenu`, utile pour reformuler une question elliptique
        ('et pour Paris ?') ou pour l'injecter dans un prompt génératif."""
        msgs = [m for m in _STORE[self.conversation_id] if m["role"] != "system"][-n_last:]
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    def clear(self) -> None:
        _STORE[self.conversation_id] = []


def get_memory(conversation_id: str = None, max_turns: int = 6) -> ConversationMemory:
    """Point d'entrée pratique pour app.py : récupère (ou crée) la
    mémoire associée à un conversation_id."""
    return ConversationMemory(conversation_id=conversation_id, max_turns=max_turns)

"""
Short-Term Memory — Per-Session Conversation Cache
===================================================

Stores recent Q&A pairs for each WebSocket session.  When a new query arrives,
we first check here using TF-IDF cosine similarity.  If a semantically similar
question was asked recently (within TTL), we return the cached context
immediately — avoiding a slower vector-store lookup.

Design decisions:
  • TF-IDF is used instead of embeddings because short-term memory is small
    (≤50 entries) and speed matters more than accuracy here.
  • Each session has isolated memory — user A's cache never leaks to user B.
  • Entries expire after `ttl_minutes` to keep memory fresh.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("voice-server")


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single Q&A pair stored in short-term memory."""
    query: str
    response: str
    context: str          # The document context that was used (if any)
    timestamp: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.time() - self.timestamp) > ttl_seconds


# ─── Short-Term Memory ──────────────────────────────────────────────────────

class ShortTermMemory:
    """
    In-process conversation cache with similarity-based retrieval.

    Usage:
        stm = ShortTermMemory(max_entries=50, ttl_minutes=30)

        # After getting a response from the LLM:
        stm.store("session-123", "What is our refund policy?", "Our refund policy...", "doc context...")

        # Before calling the LLM for a new query:
        cached = stm.search("session-123", "Tell me about refunds")
        if cached:
            print(f"Cache hit! Similarity: {cached[1]:.2f}")
            print(f"Cached response: {cached[0].response}")
    """

    def __init__(self, max_entries: int = 50, ttl_minutes: int = 30):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_minutes * 60
        # session_id → list[MemoryEntry]
        self._sessions: dict[str, list[MemoryEntry]] = {}

    def store(self, session_id: str, query: str, response: str, context: str = "") -> None:
        """Store a Q&A pair in the session's memory."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        entries = self._sessions[session_id]

        # Add the new entry
        entries.append(MemoryEntry(
            query=query,
            response=response,
            context=context,
        ))

        # Evict expired entries
        entries[:] = [e for e in entries if not e.is_expired(self.ttl_seconds)]

        # Enforce max size (remove oldest first)
        if len(entries) > self.max_entries:
            entries[:] = entries[-self.max_entries:]

        logger.info(
            f"[STM] Stored entry for session {session_id[:8]}… "
            f"(total: {len(entries)} entries)"
        )

    def search(
        self,
        session_id: str,
        query: str,
        threshold: float = 0.65,
    ) -> Optional[tuple[MemoryEntry, float]]:
        """
        Search for a similar query in the session's memory.

        Returns:
            (MemoryEntry, similarity_score) if a match above threshold is found,
            None otherwise.
        """
        if session_id not in self._sessions:
            return None

        entries = self._sessions[session_id]

        # Remove expired entries
        entries[:] = [e for e in entries if not e.is_expired(self.ttl_seconds)]

        if not entries:
            return None

        # Build corpus: all stored queries + the new query
        stored_queries = [e.query for e in entries]
        corpus = stored_queries + [query]

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(corpus)

            # Compare the new query (last row) against all stored queries
            query_vector = tfidf_matrix[-1]
            stored_vectors = tfidf_matrix[:-1]

            similarities = cosine_similarity(query_vector, stored_vectors).flatten()

            best_idx = similarities.argmax()
            best_score = similarities[best_idx]

            if best_score >= threshold:
                logger.info(
                    f"[STM] Cache HIT for session {session_id[:8]}… "
                    f"(score: {best_score:.3f}, query: '{query[:50]}…')"
                )
                return (entries[best_idx], float(best_score))

            logger.info(
                f"[STM] Cache MISS for session {session_id[:8]}… "
                f"(best score: {best_score:.3f} < threshold {threshold})"
            )

        except Exception as e:
            logger.warning(f"[STM] Search error: {e}")

        return None

    def get_recent_context(self, session_id: str, n: int = 5) -> list[MemoryEntry]:
        """
        Get the N most recent conversation entries for a session.
        Useful for providing conversation continuity to the LLM.
        """
        if session_id not in self._sessions:
            return []

        entries = self._sessions[session_id]

        # Remove expired entries
        entries[:] = [e for e in entries if not e.is_expired(self.ttl_seconds)]

        return entries[-n:]

    def clear_session(self, session_id: str) -> None:
        """Clear all memory for a specific session."""
        if session_id in self._sessions:
            count = len(self._sessions[session_id])
            del self._sessions[session_id]
            logger.info(f"[STM] Cleared {count} entries for session {session_id[:8]}…")

    def get_stats(self) -> dict:
        """Return stats about current memory usage."""
        total_entries = sum(len(v) for v in self._sessions.values())
        return {
            "active_sessions": len(self._sessions),
            "total_entries": total_entries,
            "max_entries_per_session": self.max_entries,
            "ttl_minutes": self.ttl_seconds / 60,
        }

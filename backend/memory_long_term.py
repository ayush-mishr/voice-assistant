"""
Long-Term Memory — FAISS Vector Store
======================================

Stores document embeddings in a FAISS index for semantic similarity search.
When a query isn't found in short-term memory, the Agent Core falls back here
to search through all ingested documents.

Design decisions:
  • FAISS (Facebook AI Similarity Search) is used for fast, in-memory vector
    search.  It's free and handles up to millions of vectors easily.
  • We use L2 (Euclidean distance) as the similarity metric — standard for
    dense embeddings from models like Titan Embed.
  • The index + metadata are saved to disk and can be backed up to S3 for
    persistence across ECS task restarts.
  • Each vector is associated with metadata: original text chunk, source
    filename, page number, chunk index.
"""

import os
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger("voice-server")

# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """Metadata for a single text chunk stored in the vector index."""
    text: str               # The actual text content
    source: str             # Original filename (e.g., "company_policy.pdf")
    page: int = 0           # Page number (for PDFs)
    chunk_index: int = 0    # Position within the document
    doc_id: str = ""        # Unique document ID


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk: DocumentChunk
    score: float            # Similarity score (lower L2 distance = more similar)
    rank: int               # Position in results (0 = best match)


# ─── Long-Term Memory ───────────────────────────────────────────────────────

class LongTermMemory:
    """
    FAISS-based vector store for document retrieval.

    Usage:
        ltm = LongTermMemory(dimension=1024, index_path="./faiss_data")

        # After generating embeddings for document chunks:
        ltm.add_documents(chunks, embeddings)

        # When searching:
        results = ltm.search(query_embedding, top_k=5)
        for r in results:
            print(f"Score: {r.score:.4f} | Source: {r.chunk.source}")
            print(f"  Text: {r.chunk.text[:100]}...")
    """

    def __init__(self, dimension: int = 1024, index_path: str = "./faiss_data"):
        """
        Args:
            dimension: Embedding vector dimension (1024 for Titan Embed v2).
            index_path: Directory to save/load the FAISS index and metadata.
        """
        self.dimension = dimension
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "index.faiss")
        self.metadata_file = os.path.join(index_path, "metadata.json")

        self.index = None

        # Parallel array of metadata (same order as vectors in the index)
        self.chunks: list[DocumentChunk] = []
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the index if not already loaded."""
        if not self._loaded:
            self._load()
            self._loaded = True

    def _get_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            logger.warning("[LTM] faiss-cpu not installed. Long-term memory disabled.")
            return None


    def add_documents(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add document chunks with their embeddings to the index.
        """
        self._ensure_loaded()
        faiss = self._get_faiss()
        if faiss is None:
            logger.error("[LTM] FAISS not available. Cannot add documents.")
            return 0
            
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have same length"
            )

        if not chunks:
            return self.index.ntotal

        # Convert embeddings to numpy array
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for better cosine similarity behavior with L2
        faiss.normalize_L2(vectors)

        # Add to FAISS index
        self.index.add(vectors)

        # Store metadata
        self.chunks.extend(chunks)

        # Persist to disk
        self._save()

        logger.info(
            f"[LTM] Added {len(chunks)} chunks from '{chunks[0].source}'. "
            f"Total vectors: {self.index.ntotal}"
        )

        return self.index.ntotal

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 1.5,
    ) -> list[SearchResult]:
        """
        Search for similar document chunks.
        """
        self._ensure_loaded()
        faiss = self._get_faiss()
        if faiss is None or self.index is None or self.index.ntotal == 0:
            logger.info("[LTM] Index is empty or unavailable. No results.")
            return []

        # Convert to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search FAISS index
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue  # FAISS returns -1 for empty slots

            if distance > score_threshold:
                continue  # Too dissimilar

            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(distance),
                rank=rank,
            ))

        if results:
            logger.info(
                f"[LTM] Found {len(results)} results "
                f"(best score: {results[0].score:.4f}, source: '{results[0].chunk.source}')"
            )
        else:
            logger.info("[LTM] No results above threshold.")

        return results

    def delete_document(self, doc_id: str) -> int:
        """
        Delete all chunks belonging to a specific document.
        """
        self._ensure_loaded()
        faiss = self._get_faiss()
        if faiss is None or self.index is None:
            return 0

        # Find indices to keep
        keep_indices = [
            i for i, chunk in enumerate(self.chunks)
            if chunk.doc_id != doc_id
        ]

        removed_count = len(self.chunks) - len(keep_indices)

        if removed_count == 0:
            logger.info(f"[LTM] No chunks found for doc_id '{doc_id}'")
            return 0

        if not keep_indices:
            # All chunks belong to this document — reset everything
            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks = []
            self._save()
            logger.info(f"[LTM] Removed all {removed_count} chunks (index now empty)")
            return removed_count

        # Reconstruct vectors for kept entries
        all_vectors = np.zeros((self.index.ntotal, self.dimension), dtype=np.float32)
        for i in range(self.index.ntotal):
            all_vectors[i] = self.index.reconstruct(i)

        kept_vectors = all_vectors[keep_indices]
        kept_chunks = [self.chunks[i] for i in keep_indices]

        # Rebuild index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(kept_vectors)
        self.chunks = kept_chunks

        self._save()
        logger.info(
            f"[LTM] Removed {removed_count} chunks for doc '{doc_id}'. "
            f"Remaining: {self.index.ntotal}"
        )

        return removed_count

    def list_documents(self) -> list[dict]:
        """Return a summary of all indexed documents."""
        self._ensure_loaded()
        doc_map: dict[str, dict] = {}

        for chunk in self.chunks:
            if chunk.doc_id not in doc_map:
                doc_map[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "chunk_count": 0,
                }
            doc_map[chunk.doc_id]["chunk_count"] += 1

        return list(doc_map.values())

    def get_all_chunks(self) -> list[DocumentChunk]:
        """
        Return ALL document chunks stored in the index.

        Used by AgentCore.build_session_prompt() to inject the full
        knowledge base into Nova Sonic's system prompt at session start.
        """
        self._ensure_loaded()
        return list(self.chunks)

    def get_stats(self) -> dict:
        """Return stats about the vector store."""
        self._ensure_loaded()
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(set(c.doc_id for c in self.chunks)),
            "dimension": self.dimension,
            "index_path": self.index_path,
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        """Save FAISS index and metadata to disk."""
        faiss = self._get_faiss()
        if faiss is None or self.index is None:
            return

        os.makedirs(self.index_path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_file)

        # Save metadata as JSON
        metadata = [asdict(chunk) for chunk in self.chunks]
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"[LTM] Saved index ({self.index.ntotal} vectors) to {self.index_path}")

    def _load(self) -> None:
        """Load FAISS index and metadata from disk."""
        faiss = self._get_faiss()
        if faiss is None:
            return

        if not os.path.exists(self.index_file) or not os.path.exists(self.metadata_file):
            logger.info("[LTM] No existing index found. Starting fresh.")
            return

        try:
            self.index = faiss.read_index(self.index_file)

            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.chunks = [DocumentChunk(**m) for m in metadata]

            logger.info(
                f"[LTM] Loaded index ({self.index.ntotal} vectors, "
                f"{len(set(c.doc_id for c in self.chunks))} documents) from {self.index_path}"
            )

        except Exception as e:
            logger.error(f"[LTM] Failed to load index: {e}. Starting fresh.")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.chunks = []

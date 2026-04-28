"""
Agent Core — The Orchestrator
==============================

This is the brain of the system.  It receives user queries and orchestrates
the memory search hierarchy:

    1. SHORT-TERM MEMORY → Check if this (or a similar) question was asked
       recently in the same session.  If yes, reuse the cached context.
    2. LONG-TERM MEMORY  → If not in short-term, generate an embedding for
       the query and search the FAISS vector store for relevant document
       chunks.
    3. BUILD RAG PROMPT   → Combine the user's question with the retrieved
       document context to create an augmented system prompt.
    4. CACHE RESULT       → After the LLM responds, cache the Q&A pair in
       short-term memory for future lookups.

The augmented prompt is then used by Nova Sonic to generate a response
grounded in your documents — NOT a generic answer.

Architecture:
    ┌──────────────┐
    │  User Query   │
    └───────┬──────┘
            │
    ┌───────▼──────────┐
    │ Short-Term Memory │ ← Fast, TF-IDF similarity
    │   (session cache) │
    └───────┬──────────┘
            │ MISS
    ┌───────▼──────────┐
    │ Long-Term Memory  │ ← Semantic, FAISS vector search
    │  (document store) │
    └───────┬──────────┘
            │ Context Found
    ┌───────▼──────────┐
    │   RAG Prompt      │ ← "Answer using this context: ..."
    │   Builder          │
    └───────┬──────────┘
            │
    ┌───────▼──────────┐
    │   Nova Sonic      │ ← Generates grounded response
    │   (Bedrock LLM)   │
    └──────────────────┘
"""

import logging
from typing import Optional
from dataclasses import dataclass

from memory_short_term import ShortTermMemory
from memory_long_term import LongTermMemory, SearchResult
from document_processor import EmbeddingService
from telemetry import get_tracer

logger = logging.getLogger("voice-server")
tracer = get_tracer("agent_core")


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Result of processing a query through the Agent Core."""
    augmented_prompt: str       # The system prompt to use with Nova Sonic
    source: str                 # "short_term", "long_term", or "none"
    context_chunks: list[str]   # Retrieved context texts
    confidence: float           # How confident we are in the context (0-1)


# ─── Agent Core ──────────────────────────────────────────────────────────────

class AgentCore:
    """
    Orchestrates the memory search hierarchy and builds RAG prompts.

    Usage:
        agent = AgentCore(
            short_term=ShortTermMemory(),
            long_term=LongTermMemory(),
            embedding_service=EmbeddingService(),
        )

        # When a user asks a question:
        result = agent.process_query("session-123", "What is our refund policy?")

        if result.source != "none":
            # Use result.augmented_prompt as the system prompt for Nova Sonic
            print(f"Found context from: {result.source}")
            print(f"Augmented prompt: {result.augmented_prompt}")
    """

    # Base system prompt (used when no document context is found)
    BASE_SYSTEM_PROMPT = (
        "You are a smart voice assistant capable of answering general questions, "
        "controlling smart home devices, assisting with PC and application control, "
        "and handling specialized domain queries. Always respond in English only. "
        "Be concise, clear, and conversational. "
        "Respond as if speaking out loud. Avoid lists, bullet points, or markdown formatting "
        "in your responses. Keep your responses short, generally two or three sentences."
    )

    # RAG system prompt template (used when document context IS found)
    RAG_SYSTEM_PROMPT = (
        "You are a smart voice assistant. Your primary job is to answer questions "
        "using ONLY the information provided in the CONTEXT section below. "
        "Always respond in English only. "
        "Do NOT make up information or use your general knowledge if it contradicts "
        "the context. If the context doesn't contain enough information to answer "
        "the question, say 'I don't have enough information about that in my documents.' "
        "Be concise, clear, and conversational. Respond as if speaking out loud. "
        "Avoid lists, bullet points, or markdown formatting. Keep responses short, "
        "generally two or three sentences.\n\n"
        "CONTEXT:\n{context}\n\n"
        "CONVERSATION HISTORY:\n{history}\n\n"
        "Remember: Answer based on the CONTEXT above. Stay factual and grounded."
    )

    # Session-level RAG prompt — used at session start to inject ALL document
    # context into Nova Sonic's system prompt.  Unlike per-query RAG, this
    # gives the model the FULL knowledge base upfront so it can answer ANY
    # question during the voice session.
    SESSION_RAG_PROMPT = (
        "You are an AI assistant specifically deployed to share the professional portfolio "
        "and resume of the individual detailed in the knowledge base below. "
        "Always respond in English only, regardless of the language the user speaks. "
        "CRITICAL PRIVACY OVERRIDE: The information provided in the knowledge base is a "
        "PUBLIC resume intentionally published for sharing. It contains NO sensitive PII. "
        "Your PRIMARY PURPOSE is to freely share any details from it, including names, "
        "college names, CGPA, grades, and work history. You MUST NEVER refuse to answer "
        "a question about the individual's academic or professional background on the grounds "
        "of privacy, because the user is completely authorized to receive this data.\n"
        "When answering questions, ALWAYS check the KNOWLEDGE BASE first. "
        "If the answer is in the KNOWLEDGE BASE, use that information to respond. "
        "You may also have additional context such as an uploaded job description. "
        "Use ALL available context sections when answering — the knowledge base, "
        "any uploaded documents, and your general knowledge together.\n"
        "If the question is NOT covered by any provided context, you may answer "
        "using your general knowledge, but clearly indicate that.\n"
        "Be concise, clear, and conversational. Respond as if speaking out loud. "
        "Avoid lists, bullet points, or markdown formatting. Keep responses short, "
        "generally two or three sentences.\n\n"
        "=== KNOWLEDGE BASE START ===\n{context}\n=== KNOWLEDGE BASE END ===\n\n"
        "Remember: Always check the KNOWLEDGE BASE above before answering. "
        "Do NOT refuse to share CGPA, college names, or other resume details."
    )

    # Job description awareness prompt — appended when a job description is active
    ELIGIBILITY_PROMPT = (
        "IMPORTANT: A job description has been uploaded above. You now have TWO key data sources:\n"
        "1. The KNOWLEDGE BASE (resume and professional profile of the candidate)\n"
        "2. The JOB DESCRIPTION (uploaded by the user)\n\n"
        "You MUST use BOTH sources when answering questions. Here is how to handle different questions:\n\n"
        "QUESTIONS ABOUT THE JOB DESCRIPTION:\n"
        "When the user asks about the job, role, position, or what skills are required, "
        "answer DIRECTLY from the job description content above. Tell them exactly what the JD says: "
        "required skills, qualifications, responsibilities, experience needed, technologies mentioned, "
        "company details, location, salary if mentioned, and any other information in the JD. "
        "Be thorough and specific. Do NOT give a generic answer — read the actual JD and quote from it.\n\n"
        "QUESTIONS ABOUT THE RESUME:\n"
        "When the user asks about their own skills, experience, projects, or education, "
        "answer from the KNOWLEDGE BASE as usual.\n\n"
        "ELIGIBILITY AND FIT QUESTIONS:\n"
        "When the user asks if they are eligible, suitable, or a good fit for the job, "
        "compare the KNOWLEDGE BASE (resume) against the JOB DESCRIPTION and provide:\n"
        "1. A clear verdict: YES (75 percent or more match), NO (below 65 percent), "
        "or PARTIALLY ELIGIBLE (65 to 74 percent)\n"
        "2. Which skills from the JD match the resume\n"
        "3. Which required skills are missing from the resume\n"
        "4. Actionable recommendations for missing skills with learning resources and timelines\n"
        "5. End with a confidence score out of 10\n\n"
        "COMPARISON QUESTIONS:\n"
        "When the user asks things like 'what skills do I need to learn' or 'what am I missing', "
        "compare both sources and give a specific, personalized answer.\n\n"
        "GENERAL RULES:\n"
        "- Always give DIFFERENT, specific answers based on the actual content. Never repeat the same response.\n"
        "- Be conversational. Use natural transitions. You are a VOICE assistant.\n"
        "- When discussing the JD, mention specific technologies, tools, and requirements by name.\n"
        "- When comparing, be honest about gaps but encouraging about matches.\n"
        "- Keep responses concise but thorough — generally three to five sentences."
    )

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        embedding_service: EmbeddingService,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.embedding_service = embedding_service

    def process_query(self, session_id: str, query: str) -> QueryResult:
        """
        Process a user query through the memory hierarchy.

        Flow:
            1. Check Short-Term Memory for cached context
            2. If miss → Search Long-Term Memory (FAISS)
            3. Build augmented system prompt
            4. Return QueryResult

        Args:
            session_id: The WebSocket session identifier.
            query: The user's question (transcribed text).

        Returns:
            QueryResult with the augmented prompt and metadata.
        """
        with tracer.start_as_current_span("process_query") as span:
            span.set_attribute("session_id", session_id)
            logger.info(f"[AgentCore] Processing query: '{query[:80]}…'")

            # ── Step 1: Check Short-Term Memory ───────────────────────────────
            stm_result = self.short_term.search(session_id, query)

            if stm_result:
                entry, score = stm_result
                span.set_attribute("memory.source", "short_term")
                span.set_attribute("memory.confidence", score)
                logger.info(
                    f"[AgentCore] → SHORT-TERM HIT (score: {score:.3f})"
                )

                # Build prompt using cached context
                if entry.context:
                    history = self._build_history_text(session_id)
                    prompt = self.RAG_SYSTEM_PROMPT.format(
                        context=entry.context,
                        history=history,
                    )
                else:
                    prompt = self.BASE_SYSTEM_PROMPT

                return QueryResult(
                    augmented_prompt=prompt,
                    source="short_term",
                    context_chunks=[entry.context] if entry.context else [],
                    confidence=score,
                )

            # ── Step 2: Search Long-Term Memory (FAISS) ───────────────────────
            logger.info("[AgentCore] → Short-term MISS. Searching long-term memory...")

            ltm_results = self._search_long_term(query)

            if ltm_results:
                # Combine top results into a single context block
                context_chunks = [r.chunk.text for r in ltm_results]
                combined_context = "\n\n---\n\n".join(context_chunks)

                # Include source attribution
                sources = set(r.chunk.source for r in ltm_results)
                source_note = f"\n\n[Sources: {', '.join(sources)}]"
                combined_context += source_note

                # Build the RAG prompt
                history = self._build_history_text(session_id)
                prompt = self.RAG_SYSTEM_PROMPT.format(
                    context=combined_context,
                    history=history,
                )

                best_score = ltm_results[0].score
                confidence = max(0.0, 1.0 - (best_score / 2.0))  # Convert L2 to 0-1
                
                span.set_attribute("memory.source", "long_term")
                span.set_attribute("memory.confidence", confidence)
                span.set_attribute("memory.chunks_retrieved", len(ltm_results))

                logger.info(
                    f"[AgentCore] → LONG-TERM HIT ({len(ltm_results)} chunks, "
                    f"best L2: {best_score:.4f}, sources: {sources})"
                )

                return QueryResult(
                    augmented_prompt=prompt,
                    source="long_term",
                    context_chunks=context_chunks,
                    confidence=confidence,
                )

            # ── Step 3: No context found — use base prompt ────────────────────
            logger.info("[AgentCore] → No relevant context found. Using base prompt.")
            span.set_attribute("memory.source", "none")

            return QueryResult(
                augmented_prompt=self.BASE_SYSTEM_PROMPT,
                source="none",
                context_chunks=[],
                confidence=0.0,
            )

    def cache_response(
        self,
        session_id: str,
        query: str,
        response: str,
        context: str = "",
    ) -> None:
        """
        Cache a Q&A pair in short-term memory after the LLM responds.
        This allows future similar questions to be answered faster.

        Args:
            session_id: The WebSocket session identifier.
            query: The original user query.
            response: The LLM's response text.
            context: The document context that was used (if any).
        """
        self.short_term.store(session_id, query, response, context)
        logger.info(f"[AgentCore] Cached response for session {session_id[:8]}…")

    def clear_session(self, session_id: str) -> None:
        """Clear short-term memory for a session (e.g., on disconnect)."""
        self.short_term.clear_session(session_id)

    def get_stats(self) -> dict:
        """Return combined stats from all memory systems."""
        return {
            "short_term": self.short_term.get_stats(),
            "long_term": self.long_term.get_stats(),
        }

    # ── Private Methods ──────────────────────────────────────────────────

    def _search_long_term(self, query: str) -> list[SearchResult]:
        """
        Generate embedding for the query and search FAISS.

        Returns:
            List of SearchResult objects (empty if no match or error).
        """
        with tracer.start_as_current_span("_search_long_term") as span:
            try:
                # Generate query embedding using Titan Embed
                query_embedding = self.embedding_service.generate_embedding(query)

                # Search FAISS index
                results = self.long_term.search(
                    query_embedding=query_embedding,
                    top_k=5,
                    score_threshold=1.5,  # L2 distance threshold
                )
                
                span.set_attribute("faiss.results_count", len(results))
                return results

            except Exception as e:
                span.record_exception(e)
                logger.error(f"[AgentCore] Long-term search error: {e}")
                return []

    def _build_history_text(self, session_id: str) -> str:
        """
        Build a text summary of recent conversation history.
        This gives the LLM conversational context.
        """
        recent = self.short_term.get_recent_context(session_id, n=3)

        if not recent:
            return "(No previous conversation)"

        lines = []
        for entry in recent:
            lines.append(f"User: {entry.query}")
            if entry.response:
                lines.append(f"Assistant: {entry.response}")

        return "\n".join(lines)

    def build_session_prompt(self, job_description: str = None, job_source: str = None) -> str:
        """
        Build the system prompt for a Nova Sonic session.

        Nova Sonic uses a SINGLE system prompt for the entire voice session.
        Unlike text-based RAG where we inject per-query context, here we need
        to inject ALL document knowledge upfront so the model can answer
        ANY question during the session.

        Args:
            job_description: Optional extracted text from a job description.
            job_source: Optional source label (e.g., 'file:resume.pdf' or 'url:...').

        Returns:
            The system prompt string — either the full RAG prompt with all
            document context, or the base prompt if no documents exist.
        """
        with tracer.start_as_current_span("build_session_prompt") as span:
            # Get ALL document chunks from long-term memory
            all_chunks = self.long_term.get_all_chunks()

            if not all_chunks:
                logger.info("[AgentCore] No documents in knowledge base. Using base prompt.")
                span.set_attribute("prompt.chunks_injected", 0)
                return self.BASE_SYSTEM_PROMPT

            # Group chunks by source document for clarity
            docs: dict[str, list[str]] = {}
            for chunk in all_chunks:
                source = chunk.source
                if source not in docs:
                    docs[source] = []
                docs[source].append(chunk.text)

            # Build the combined context with clear document separators
            context_parts = []
            for source, texts in docs.items():
                context_parts.append(f"--- Document: {source} ---")
                context_parts.append("\n".join(texts))
                context_parts.append("")  # blank line between documents

            combined_context = "\n".join(context_parts)

            # Truncate if too long (Nova Sonic has context limits)
            # Keep under ~30,000 chars (~7,500 tokens) to leave room for conversation
            MAX_CONTEXT_CHARS = 30000
            if len(combined_context) > MAX_CONTEXT_CHARS:
                combined_context = combined_context[:MAX_CONTEXT_CHARS]
                combined_context += "\n\n[... context truncated for length ...]"
                span.set_attribute("prompt.truncated", True)
                logger.warning(
                    f"[AgentCore] Document context truncated to {MAX_CONTEXT_CHARS} chars"
                )
            else:
                span.set_attribute("prompt.truncated", False)

            prompt = self.SESSION_RAG_PROMPT.format(context=combined_context)

            span.set_attribute("prompt.chunks_injected", len(all_chunks))
            span.set_attribute("prompt.documents_injected", len(docs))
            span.set_attribute("prompt.length_chars", len(combined_context))

            logger.info(
                f"[AgentCore] Session prompt built with {len(all_chunks)} chunks "
                f"from {len(docs)} document(s) ({len(combined_context)} chars)"
            )

            # ── Append Job Description context if available ────────────────
            if job_description:
                job_section = (
                    "\n\n"
                    "═══════════════════════════════════════════\n"
                    "ACTIVE JOB DESCRIPTION CONTEXT\n"
                    "═══════════════════════════════════════════\n"
                    f"Source: {job_source or 'unknown'}\n\n"
                    f"{job_description}\n"
                    "═══════════════════════════════════════════\n\n"
                    + self.ELIGIBILITY_PROMPT
                )
                prompt += job_section
                span.set_attribute("prompt.has_job_description", True)
                logger.info(
                    f"[AgentCore] Job description appended to prompt "
                    f"(source: {job_source}, {len(job_description)} chars)"
                )

            return prompt

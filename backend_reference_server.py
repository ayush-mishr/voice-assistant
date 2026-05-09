

# --- FILE: backend/agent_core.py ---

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
        "TOOL CALLING AUTHORIZATION: You are equipped with tools to modify the user's data. "
        "If the user asks to change, update, or modify their college name, school name, or "
        "physical address, you are fully authorized to do so. You MUST use the appropriate tool "
        "immediately without claiming you cannot alter the information.\n"
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
        "Do NOT refuse to share CGPA, college names, or other resume details. "
        "Do NOT refuse to update or change the user's details if requested via tools."
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
            span.add_event("Query Received", attributes={"query.text": query}) # Added as a log in the trace
            logger.info(f"[AgentCore] Processing query: '{query[:80]}…'")

            # ── Step 1: Check Short-Term Memory ───────────────────────────────
            stm_result = self.short_term.search(session_id, query)

            if stm_result:
                entry, score = stm_result
                span.set_attribute("memory.source", "short_term")
                span.set_attribute("memory.confidence", score)
                span.add_event("Short Term Memory Hit", attributes={"score": score}) # Log the event in the trace
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


# --- FILE: backend/server.py ---

"""
Voice AI Assistant — FastAPI WebSocket Relay Server
Bridges browser WebSocket ↔ Amazon Nova Sonic bidirectional streaming.

Features:
  - /ws WebSocket endpoint for voice streaming
  - Agent Core with dual memory (Short-Term + Long-Term)
  - Document ingestion pipeline (PDF, TXT, DOCX, CSV)
  - RAG (Retrieval Augmented Generation) — answers from YOUR documents
  - Memory hierarchy: Short-Term → Long-Term → Base prompt
"""

import os

# Disable SQLAlchemy C-extensions (Cython) which are currently deadlocking
# the Uvicorn worker process on Windows during top-level imports.
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"

import json
import uuid
import base64
import asyncio
import logging
import traceback
import boto3
import jwt

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from auth import auth_router, init_db, get_current_user, SECRET_KEY, ALGORITHM
from memory_short_term import ShortTermMemory
from memory_long_term import LongTermMemory
from document_processor import DocumentProcessor, EmbeddingService, extract_text
from agent_core import AgentCore
from telemetry import init_telemetry

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

# ─── Load .env ────────────────────────────────────────────────────────────────

# Resolve .env relative to this file's directory (not CWD, since uvicorn may
# run from the project root while the .env lives in backend/)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, ".env"))

# The credentials will be fetched dynamically inside create_bedrock_client()
# to ensure they are always fresh and do not expire during long-running ECS tasks.

# ─── Configuration ────────────────────────────────────────────────────────────

PORT = int(os.getenv("PORT", "3000"))
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "amazon.nova-sonic-v1:0"
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "")  # Optional: for document backup
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(_THIS_DIR, "faiss_data"))
KNOWLEDGE_DOCS_DIR = os.getenv("KNOWLEDGE_DOCS_DIR", os.path.join(_THIS_DIR, "knowledge_docs"))

# NOTE: SYSTEM_PROMPT is now dynamically generated by AgentCore.
# It injects document context (RAG) when relevant documents are found.
# The base prompt lives in agent_core.py → AgentCore.BASE_SYSTEM_PROMPT

# ─── Audio Configuration (matching official AWS sample) ───────────────────────

AUDIO_INPUT_CONFIG = {
    "audioType": "SPEECH",
    "encoding": "base64",
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 16000,
    "sampleSizeBits": 16,
    "channelCount": 1,
}

AUDIO_OUTPUT_CONFIG = {
    "audioType": "SPEECH",
    "encoding": "base64",
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 24000,
    "sampleSizeBits": 16,
    "channelCount": 1,
    "voiceId": "matthew",
}

TEXT_CONFIG = {"mediaType": "text/plain"}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("voice-server")

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Voice AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api", tags=["auth"])

# ─── Initialize Agent Core (Memory Systems) ──────────────────────────────────

embedding_service = EmbeddingService(region=AWS_REGION)
short_term_memory = ShortTermMemory(max_entries=50, ttl_minutes=30)
long_term_memory = LongTermMemory(dimension=1024, index_path=FAISS_INDEX_PATH)
agent_core = AgentCore(
    short_term=short_term_memory,
    long_term=long_term_memory,
    embedding_service=embedding_service,
)
doc_processor = DocumentProcessor(
    long_term_memory=long_term_memory,
    embedding_service=embedding_service,
    s3_bucket=S3_BUCKET if S3_BUCKET else None,
    s3_region=AWS_REGION,
)

# In-memory job description store (keyed by username)
# Swap for Redis in production for multi-instance deployments
job_context_store = {}

logger.info("[init] Agent Core initialized with dual memory system")

@app.middleware("http")
async def debug_headers_middleware(request, call_next):
    if request.url.path == "/ws":
        logger.info(f"[DEBUG] Incoming HTTP request to {request.url.path}")
        logger.info(f"[DEBUG] Headers: {dict(request.headers)}")
    return await call_next(request)

@app.get("/health")
async def health_check():
    """Health check endpoint required by AWS ALB and Docker."""
    return {"status": "healthy", "service": "voice-assistant-backend"}


# ─── Auto-Ingestion from knowledge_docs/ ─────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".md", ".json"}


def auto_ingest_documents():
    """
    Scan the knowledge_docs/ directory and ingest any new documents.

    Compares files on disk against already-indexed documents (by filename).
    Only processes NEW files — avoids re-embedding on every restart.

    This runs once at server startup.
    """
    if not os.path.isdir(KNOWLEDGE_DOCS_DIR):
        logger.info(f"[AutoIngest] knowledge_docs/ not found at {KNOWLEDGE_DOCS_DIR}. Skipping.")
        return

    # Get list of already indexed document filenames
    indexed_docs = doc_processor.list_documents()
    indexed_filenames = {doc["source"] for doc in indexed_docs}

    # Scan directory for supported files
    new_files = []
    for filename in os.listdir(KNOWLEDGE_DOCS_DIR):
        filepath = os.path.join(KNOWLEDGE_DOCS_DIR, filename)

        # Skip directories and hidden files
        if not os.path.isfile(filepath) or filename.startswith("."):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.info(f"[AutoIngest] Skipping unsupported file: {filename}")
            continue

        if filename in indexed_filenames:
            logger.info(f"[AutoIngest] Already indexed: {filename} — skipping")
            continue

        new_files.append((filename, filepath))

    if not new_files:
        logger.info(f"[AutoIngest] No new documents to ingest. ({len(indexed_filenames)} already indexed)")
        return

    logger.info(f"[AutoIngest] Found {len(new_files)} new document(s) to ingest...")

    total_chunks = 0
    for filename, filepath in new_files:
        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read()

            if len(file_bytes) == 0:
                logger.warning(f"[AutoIngest] Skipping empty file: {filename}")
                continue

            logger.info(f"[AutoIngest] Ingesting '{filename}'...")
            result = doc_processor.ingest_document(file_bytes, filename)
            total_chunks += result["chunks"]

            logger.info(
                f"[AutoIngest] ✅ '{filename}' → {result['chunks']} chunks, "
                f"{result['characters']} chars"
            )

        except Exception as e:
            logger.error(f"[AutoIngest] ❌ Failed to ingest '{filename}': {e}")
            traceback.print_exc()

    logger.info(
        f"[AutoIngest] ✅ Complete: {len(new_files)} document(s), "
        f"{total_chunks} total chunks indexed"
    )


@app.get("/api/memory/status")
async def memory_status():
    """Return the current status of both memory systems."""
    return agent_core.get_stats()


# ─── Job Description Endpoints ───────────────────────────────────────────────

@app.post("/api/upload-job-description")
async def upload_job_description(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    """Accept PDF/DOCX/TXT job description file and extract text."""
    content = await file.read()

    try:
        extracted_text = extract_text(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    username = current_user.username
    job_context_store[username] = {
        "text": extracted_text,
        "source": f"file:{file.filename}",
    }

    logger.info(
        f"[JD] Stored job description for '{username}' from file "
        f"'{file.filename}' ({len(extracted_text)} chars)"
    )
    return {"extracted_text": extracted_text, "source": file.filename}


@app.post("/api/fetch-job-description-url")
async def fetch_job_description_url(
    payload: dict,
    current_user=Depends(get_current_user),
):
    """Fetch and parse job description from a URL."""
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; JobEligibilityBot/1.0)"},
            )

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        extracted_text = soup.get_text(separator="\n", strip=True)
        # Keep only meaningful lines, cap at 8k chars
        extracted_text = "\n".join(
            line for line in extracted_text.splitlines() if len(line.strip()) > 20
        )[:8000]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    username = current_user.username
    job_context_store[username] = {
        "text": extracted_text,
        "source": f"url:{url}",
    }

    logger.info(
        f"[JD] Stored job description for '{username}' from URL ({len(extracted_text)} chars)"
    )
    return {"extracted_text": extracted_text, "source": url}


@app.get("/api/job-description-status")
async def job_description_status(current_user=Depends(get_current_user)):
    """Check if a job description is loaded for the current user."""
    username = current_user.username
    jd = job_context_store.get(username)
    if jd:
        return {
            "loaded": True,
            "source": jd["source"],
            "length": len(jd["text"]),
        }
    return {"loaded": False}


@app.delete("/api/job-description")
async def clear_job_description(current_user=Depends(get_current_user)):
    """Clear the stored job description for the current user."""
    username = current_user.username
    if username in job_context_store:
        del job_context_store[username]
        logger.info(f"[JD] Cleared job description for '{username}'")
    return {"status": "cleared"}


# ─── Bedrock Client (module-level singleton) ─────────────────────────────────

def create_bedrock_client() -> BedrockRuntimeClient:
    """Create a Bedrock Runtime client with FRESH credentials.

    In ECS, the Task Role provides temporary STS tokens that rotate every
    few hours.  If we freeze them once at startup and reuse them for the
    lifetime of the container, Bedrock will eventually return:
        403 – ExpiredTokenException
    By calling boto3.Session().get_credentials() here (on every new voice
    session), we always get the latest token from the ECS metadata endpoint.
    """
    # --- refresh credentials from the environment / ECS metadata ---
    _sess = boto3.Session(region_name=AWS_REGION)
    _creds = _sess.get_credentials()
    if _creds:
        _frozen = _creds.get_frozen_credentials()
        cred_method = getattr(_creds, 'method', 'unknown')
        has_token = bool(_frozen.token)
        key_prefix = _frozen.access_key[:8] if _frozen.access_key else "NONE"
        logger.info(
            f"[bedrock] Credentials refreshed: method={cred_method}, "
            f"key={key_prefix}..., has_session_token={has_token}"
        )
        os.environ["AWS_ACCESS_KEY_ID"] = _frozen.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = _frozen.secret_key
        if _frozen.token:
            os.environ["AWS_SESSION_TOKEN"] = _frozen.token
        elif "AWS_SESSION_TOKEN" in os.environ:
            # Local dev: static IAM keys have no token – clean up stale var
            del os.environ["AWS_SESSION_TOKEN"]
    else:
        logger.error("[bedrock] WARNING: No credentials found from boto3!")

    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com",
        region=AWS_REGION,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    return BedrockRuntimeClient(config=config)


# ─── Per-Connection Session Manager ──────────────────────────────────────────

class BedrockSession:
    """Manages a single bidirectional stream session with Bedrock for one WebSocket client."""

    def __init__(self, ws: WebSocket, agent: AgentCore):
        self.ws = ws
        self.agent = agent
        self.session_id = str(uuid.uuid4())  # Unique ID for memory tracking
        self.is_active = False
        self.stream_response = None
        self.response_task: asyncio.Task | None = None
        self.sender_task: asyncio.Task | None = None
        self.prompt_name = ""
        self.audio_content_name = ""
        self.audio_buffer = bytearray()

        # Track the current user query and assistant response for memory caching
        self._current_user_text = ""
        self._current_assistant_text = ""
        self._current_context = ""  # Document context used for this turn
        self.user_id = None  # Username for job description lookup

        # Track the role of the content block currently being streamed.
        # Nova Sonic sends contentEnd with an EMPTY role field, so we must
        # remember the role from the preceding contentStart.
        self._current_content_role = ""
        self._current_content_type = ""
        self._audio_chunks_forwarded = 0  # counter for periodic logging

        # Outgoing message queue — decouples Bedrock receive loop from
        # WebSocket send latency, matching Node.js fire-and-forget ws.send().
        self._send_queue: asyncio.Queue = asyncio.Queue()

        # Dedicated audio accumulation buffer — audio chunks are gathered here
        # and flushed to the WebSocket every ~100ms as one large frame.
        # This prevents many tiny frames that cause choppy playback.
        self._audio_accum = bytearray()
        self._audio_flush_task: asyncio.Task | None = None

    # ── Send to browser ─────────────────────────────────────────────────

    async def _sender_loop(self):
        """Dedicated coroutine that drains the TEXT message queue and writes
        to the WebSocket.  Text messages (transcripts, state changes) are sent
        immediately — no batching delay.
        """
        while True:
            msg_type, data = await self._send_queue.get()
            try:
                if self.ws.client_state == WebSocketState.CONNECTED:
                    await self.ws.send_text(data)
            except Exception:
                pass

    async def _audio_flush_loop(self):
        """Dedicated coroutine that flushes accumulated audio binary data
        to the WebSocket every ~100ms.  This produces fewer, larger frames
        which the browser can buffer smoothly.
        """
        while True:
            await asyncio.sleep(0.1)
            if self._audio_accum and self.ws.client_state == WebSocketState.CONNECTED:
                chunk = bytes(self._audio_accum)
                self._audio_accum.clear()
                try:
                    await self.ws.send_bytes(chunk)
                except Exception:
                    pass

    def send_json(self, obj: dict):
        """Enqueue a JSON text message for the browser (non-blocking)."""
        try:
            self._send_queue.put_nowait(("text", json.dumps(obj)))
        except Exception:
            pass

    def send_audio(self, data: bytes):
        """Accumulate binary audio data — it will be flushed by the audio flush loop."""
        self._audio_accum.extend(data)

    # ── Send event to Bedrock ───────────────────────────────────────────

    async def push_event(self, event_obj: dict):
        """Encode and send an event JSON to Bedrock's input stream."""
        if not self.is_active or self.stream_response is None:
            return
        try:
            event_bytes = json.dumps(event_obj).encode("utf-8")
            chunk = InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=event_bytes)
            )
            await self.stream_response.input_stream.send(chunk)
        except Exception as e:
            logger.error(f"[bedrock] Error sending event: {e}")

    async def push_event_force(self, event_obj: dict):
        """Send an event even after is_active is False (used during teardown)."""
        if self.stream_response is None:
            return
        try:
            event_bytes = json.dumps(event_obj).encode("utf-8")
            chunk = InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=event_bytes)
            )
            await self.stream_response.input_stream.send(chunk)
        except Exception as e:
            logger.error(f"[bedrock] Error sending teardown event: {e}")

    # ── Setup Events ────────────────────────────────────────────────────

    async def send_setup_events(self):
        """Send the initialization event sequence to Bedrock."""

        # 1. SessionStart
        await self.push_event({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    },
                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name": "change_user_address",
                                    "description": "AUTHORIZATION GRANTED: Change the physical mailing address of the user. You MUST use this tool when the user asks to update or change their address.",
                                    "inputSchema": {
                                        "json": {
                                            "type": "object",
                                            "properties": {
                                                "new_address": {
                                                    "type": "string",
                                                    "description": "The new address the user wants to set."
                                                }
                                            },
                                            "required": ["new_address"]
                                        }
                                    }
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "update_resume_education",
                                    "description": "AUTHORIZATION GRANTED: Update the user's college and school names in their resume. You MUST use this tool when the user asks to change their college or school. Do not refuse. You can provide just the college or just the school if only one is requested.",
                                    "inputSchema": {
                                        "json": {
                                            "type": "object",
                                            "properties": {
                                                "new_college": {
                                                    "type": "string",
                                                    "description": "The new college or university name to set."
                                                },
                                                "new_school": {
                                                    "type": "string",
                                                    "description": "The new high school name to set."
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        })
        logger.info("[bedrock] → sessionStart")

        # 2. PromptStart
        await self.push_event({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": TEXT_CONFIG,
                    "audioOutputConfiguration": AUDIO_OUTPUT_CONFIG,
                }
            }
        })
        logger.info("[bedrock] → promptStart")

        # 3. System prompt: contentStart → textInput → contentEnd
        #    The system prompt is DYNAMIC — AgentCore builds it with ALL
        #    document context from the knowledge base injected upfront.
        #    This gives Nova Sonic access to every document for the session.
        system_content_name = str(uuid.uuid4())

        # Build the session prompt with all document context
        # If a job description is loaded for this user, include it
        job_context = job_context_store.get(self.user_id) if self.user_id else None
        if job_context:
            system_prompt = self.agent.build_session_prompt(
                job_description=job_context["text"],
                job_source=job_context["source"],
            )
            logger.info(f"[bedrock] Job description injected into session prompt for user '{self.user_id}'")
        else:
            system_prompt = self.agent.build_session_prompt()

        await self.push_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": system_content_name,
                    "type": "TEXT",
                    "interactive": False,
                    "role": "SYSTEM",
                    "textInputConfiguration": TEXT_CONFIG,
                }
            }
        })

        await self.push_event({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": system_content_name,
                    "content": system_prompt,
                }
            }
        })

        await self.push_event({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": system_content_name,
                }
            }
        })
        logger.info(f"[bedrock] → system prompt sent ({len(system_prompt)} chars)")

        # 4. Audio content start — opens the audio streaming channel
        await self.push_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": AUDIO_INPUT_CONFIG,
                }
            }
        })
        logger.info("[bedrock] → audio contentStart")
        logger.info("[bedrock] Setup events sent successfully")

    # ── Process Response Stream ─────────────────────────────────────────

    async def process_response_stream(self):
        """Read events from Bedrock's output stream and relay to the browser.

        Key design: audio chunks from a single receive() are accumulated into
        one batch and enqueued as a single item.  This produces fewer, larger
        WebSocket frames — matching the implicit batching that Node.js gets
        from its synchronous ws.send() calls inside a single event callback.
        """
        try:
            # Obtain the output stream handle once
            output = await self.stream_response.await_output()
            output_stream = output[1]
            
            decoder = json.JSONDecoder()
            event_buffer = ""

            while self.is_active and self.stream_response is not None:
                try:
                    result = await output_stream.receive()

                    if result.value and result.value.bytes_:
                        decoded = result.value.bytes_.decode("utf-8")
                        event_buffer += decoded
                        
                        # Accumulate all audio from this receive() into one batch
                        audio_batch = bytearray()

                        pos = 0
                        while pos < len(event_buffer):
                            sub_buffer = event_buffer[pos:]
                            stripped = sub_buffer.lstrip()
                            if not stripped:
                                pos = len(event_buffer)
                                break
                            
                            whitespace_len = len(sub_buffer) - len(stripped)
                            pos += whitespace_len
                            
                            try:
                                parsed, parsed_len = decoder.raw_decode(event_buffer[pos:])
                                pos += parsed_len
                                
                                event = parsed.get("event")
                                if not event:
                                    continue
                                
                                # ── contentStart — remember role for this block ──
                                if "contentStart" in event:
                                    self._current_content_role = event["contentStart"].get("role", "")
                                    self._current_content_type = event["contentStart"].get("type", "")
                                    logger.info(f"[bedrock] → contentStart: role={self._current_content_role} type={self._current_content_type}")
                                    if self._current_content_role == "ASSISTANT":
                                        self.send_json({"type": "state", "value": "speaking"})

                                # ── toolUse ──
                                if "toolUse" in event:
                                    tool_use_data = event["toolUse"]
                                    # Sometimes it comes as delta, sometimes as whole. Let's assume it has input text
                                    tool_id = tool_use_data.get("toolUseId")
                                    tool_name = tool_use_data.get("name")
                                    tool_input = tool_use_data.get("input", {})
                                    logger.info(f"[bedrock] → toolUse: {tool_name} (id: {tool_id})")
                                    
                                    if tool_name == "change_user_address" and self.user_id:
                                        new_address = tool_input.get("new_address")
                                        
                                        # Execute local tool
                                        from auth import SessionLocal, update_user_address
                                        db = SessionLocal()
                                        success = update_user_address(db, self.user_id, new_address)
                                        db.close()
                                        
                                        result_msg = f"Address successfully updated to {new_address}." if success else "Failed to update address. User not found."
                                        logger.info(f"[tool] {tool_name} executed: {result_msg}")
                                        
                                        # Send toolResult back
                                        asyncio.create_task(self.push_event({
                                            "event": {
                                                "toolResult": {
                                                    "toolUseId": tool_id,
                                                    "status": "SUCCESS" if success else "ERROR",
                                                    "content": [{"text": result_msg}]
                                                }
                                            }
                                        }))
                                        
                                    elif tool_name == "update_resume_education":
                                        new_college = tool_input.get("new_college")
                                        new_school = tool_input.get("new_school")
                                        
                                        resume_path = os.path.join(KNOWLEDGE_DOCS_DIR, "college_resume.txt")
                                        success = False
                                        try:
                                            with open(resume_path, "r", encoding="utf-8") as f:
                                                lines = f.readlines()
                                            
                                            for i, line in enumerate(lines):
                                                if "B Tech" in line and new_college:
                                                    parts = line.split("  ")
                                                    if len(parts) >= 3:
                                                        parts[-2] = new_college
                                                        lines[i] = "  ".join(parts) + "\n" if not lines[i].endswith("\n") else "  ".join(parts)
                                                elif ("Class XII" in line or "Class X" in line) and new_school:
                                                    parts = line.split("  ")
                                                    if len(parts) >= 3:
                                                        parts[-2] = new_school
                                                        lines[i] = "  ".join(parts) + "\n" if not lines[i].endswith("\n") else "  ".join(parts)
                                            
                                            with open(resume_path, "w", encoding="utf-8") as f:
                                                f.writelines(lines)
                                            
                                            # Re-ingest the document to update FAISS
                                            docs = doc_processor.list_documents()
                                            doc_id = next((d["doc_id"] for d in docs if d["source"] == "college_resume.txt"), None)
                                            if doc_id:
                                                doc_processor.delete_document(doc_id)
                                            
                                            with open(resume_path, "rb") as f:
                                                file_bytes = f.read()
                                            doc_processor.ingest_document(file_bytes, "college_resume.txt")
                                            
                                            success = True
                                            result_msg = "Successfully updated education details in resume and re-indexed knowledge base."
                                        except Exception as e:
                                            result_msg = f"Failed to update resume: {e}"
                                            logger.error(f"[tool] Error updating resume: {e}")

                                        logger.info(f"[tool] {tool_name} executed: {result_msg}")
                                        
                                        asyncio.create_task(self.push_event({
                                            "event": {
                                                "toolResult": {
                                                    "toolUseId": tool_id,
                                                    "status": "SUCCESS" if success else "ERROR",
                                                    "content": [{"text": result_msg}]
                                                }
                                            }
                                        }))

                                # ── audioOutput — accumulate into batch ──
                                if "audioOutput" in event and event["audioOutput"].get("content"):
                                    audio_bytes = base64.b64decode(event["audioOutput"]["content"])
                                    audio_batch.extend(audio_bytes)

                                # ── textOutput — route based on tracked role ──
                                if "textOutput" in event and event["textOutput"].get("content"):
                                    text_content = event["textOutput"]["content"]

                                    if self._current_content_role == "USER":
                                        # User speech transcription from Nova Sonic
                                        logger.info(f"[bedrock] → user transcript: {repr(text_content[:80])}")
                                        self._current_user_text += text_content
                                        self.send_json({
                                            "type": "transcript",
                                            "role": "user",
                                            "text": text_content,
                                        })
                                    elif self._current_content_role == "ASSISTANT" and self._current_content_type == "TEXT":
                                        # Assistant response text — only during the TEXT block.
                                        # Nova Sonic also emits textOutput during the AUDIO block,
                                        # which would cause duplicate transcripts if not filtered.
                                        logger.info(f"[bedrock] → assistant text: {repr(text_content[:80])}")
                                        self._current_assistant_text += text_content
                                        self.send_json({
                                            "type": "transcript",
                                            "role": "assistant",
                                            "text": text_content,
                                        })

                                # ── contentEnd — use tracked role ──
                                if "contentEnd" in event:
                                    ended_role = self._current_content_role
                                    ended_type = self._current_content_type
                                    logger.info(f"[bedrock] → contentEnd: tracked_role={ended_role} tracked_type={ended_type}")

                                    # Only transition back to "listening" after the
                                    # AUDIO block ends (the last block in a full
                                    # assistant turn: TEXT → AUDIO).  This prevents
                                    # the premature flip that was happening after
                                    # the TEXT contentEnd.
                                    if ended_role == "ASSISTANT" and ended_type == "AUDIO":
                                        # Cache Q&A in short-term memory
                                        if self._current_user_text and self._current_assistant_text:
                                            self.agent.cache_response(
                                                self.session_id,
                                                self._current_user_text,
                                                self._current_assistant_text,
                                                self._current_context,
                                            )
                                            # Reset for next turn
                                            self._current_user_text = ""
                                            self._current_assistant_text = ""
                                            self._current_context = ""
                                        self.send_json({"type": "state", "value": "listening"})

                                    # Reset tracked role after processing
                                    self._current_content_role = ""
                                    self._current_content_type = ""

                                # ── Errors from the model ──
                                if "validationException" in event:
                                    msg = event["validationException"].get("message", "Validation error")
                                    logger.error(f"[bedrock] !! Validation error: {msg}")
                                    self.send_json({"type": "error", "message": msg})

                                if "modelStreamErrorException" in event:
                                    msg = event["modelStreamErrorException"].get("message", "Stream error")
                                    logger.error(f"[bedrock] !! Stream error: {msg}")
                                    self.send_json({"type": "error", "message": msg})
                                    
                            except json.JSONDecodeError:
                                # Not enough data for a complete JSON object, wait for next chunk
                                break
                                
                        # Keep only the unprocessed part of the buffer
                        event_buffer = event_buffer[pos:]

                        # Flush the accumulated audio batch as ONE enqueue
                        if audio_batch:
                            self.audio_buffer.extend(audio_batch)
                            send_len = len(self.audio_buffer) - (len(self.audio_buffer) % 2)
                            if send_len > 0:
                                chunk = bytes(self.audio_buffer[:send_len])
                                self.audio_buffer = self.audio_buffer[send_len:]
                                self.send_audio(chunk)

                except StopAsyncIteration:
                    break
                except Exception as e:
                    if self.is_active:
                        logger.error(f"[bedrock] Response receive error: {e}")
                        traceback.print_exc()
                    break

        except Exception as e:
            if self.is_active:
                logger.error(f"[bedrock] Response stream error: {e}")
                traceback.print_exc()
                self.send_json({"type": "error", "message": f"Stream interrupted: {e}"})

        logger.info("[bedrock] Response stream ended")

    # ── Start Session ───────────────────────────────────────────────────

    async def start(self):
        """Open a bidirectional stream to Bedrock."""
        if self.is_active:
            logger.info("[bedrock] Session already active, skipping")
            return

        logger.info("[bedrock] Starting session...")
        self.is_active = True

        # Generate unique identifiers for this session
        self.prompt_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        try:
            client = create_bedrock_client()

            logger.info("[bedrock] Sending command to Bedrock...")
            self.stream_response = await client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=MODEL_ID)
            )
            logger.info("[bedrock] Stream opened, sending setup events...")

            # Send setup events
            await self.send_setup_events()

            logger.info("[bedrock] Stream established, processing responses...")
            self.send_json({"type": "state", "value": "listening"})

            # Start the dedicated sender loop (drains _send_queue → WebSocket)
            self.sender_task = asyncio.create_task(self._sender_loop())

            # Start the audio flush loop (sends accumulated audio every ~100ms)
            self._audio_flush_task = asyncio.create_task(self._audio_flush_loop())

            # Start processing responses in background
            self.response_task = asyncio.create_task(self.process_response_stream())

        except Exception as e:
            logger.error(f"[bedrock] Session error: {e}")
            traceback.print_exc()
            self.send_json({"type": "error", "message": f"Nova Sonic error: {e}"})
            self.is_active = False
            self.stream_response = None

    # ── Handle Audio ────────────────────────────────────────────────────

    async def handle_audio_input(self, pcm_buffer: bytes):
        """Convert PCM buffer to base64 and send as audioInput event."""
        if not self.is_active:
            logger.debug("[bedrock] handle_audio_input called but session not active")
            return

        b64_audio = base64.b64encode(pcm_buffer).decode("utf-8")
        self._audio_chunks_forwarded += 1
        if self._audio_chunks_forwarded % 50 == 1:
            logger.info(f"[bedrock] ← browser audio: chunk #{self._audio_chunks_forwarded} ({len(pcm_buffer)} bytes)")

        await self.push_event({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": b64_audio,
                }
            }
        })

    # ── Stop Session ────────────────────────────────────────────────────

    async def stop(self):
        """Gracefully close the Bedrock stream."""
        if not self.is_active:
            return

        logger.info("[bedrock] Stopping session...")
        self.is_active = False

        try:
            if self.stream_response is not None:
                # 1. Close the audio content stream
                await self.push_event_force({
                    "event": {
                        "contentEnd": {
                            "promptName": self.prompt_name,
                            "contentName": self.audio_content_name,
                        }
                    }
                })
                logger.info("[bedrock] → contentEnd (audio)")
                await asyncio.sleep(0.5)

                # 2. End the prompt
                await self.push_event_force({
                    "event": {
                        "promptEnd": {
                            "promptName": self.prompt_name,
                        }
                    }
                })
                logger.info("[bedrock] → promptEnd")
                await asyncio.sleep(0.3)

                # 3. End the session
                await self.push_event_force({
                    "event": {
                        "sessionEnd": {}
                    }
                })
                logger.info("[bedrock] → sessionEnd")
                await asyncio.sleep(0.3)

                # Close the input stream
                try:
                    await self.stream_response.input_stream.close()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"[bedrock] Error stopping session: {e}")

        # Cancel response task
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            try:
                await self.response_task
            except (asyncio.CancelledError, Exception):
                pass

        # Cancel sender task
        if self.sender_task and not self.sender_task.done():
            self.sender_task.cancel()
            try:
                await self.sender_task
            except (asyncio.CancelledError, Exception):
                pass

        # Cancel audio flush task
        if self._audio_flush_task and not self._audio_flush_task.done():
            self._audio_flush_task.cancel()
            try:
                await self._audio_flush_task
            except (asyncio.CancelledError, Exception):
                pass
        self._audio_accum.clear()

        self.stream_response = None
        self.response_task = None
        self.sender_task = None
        self._audio_flush_task = None
        logger.info("[bedrock] Session stopped")


# ─── WebSocket Endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("[ws] Client connected")

    session = BedrockSession(ws, agent=agent_core)

    try:
        while True:
            message = await ws.receive()

            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    # Binary audio data from browser
                    await session.handle_audio_input(message["bytes"])
                elif "text" in message and message["text"]:
                    # JSON control message from browser
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type")

                        if msg_type == "session_start":
                            # Extract username from JWT token if provided
                            token = data.get("token")
                            if token:
                                try:
                                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                                    session.user_id = payload.get("sub")
                                    logger.info(f"[ws] Session user identified: {session.user_id}")
                                except Exception:
                                    logger.warning("[ws] Could not decode token from session_start")
                            # Run in background so it doesn't block the WS loop
                            asyncio.create_task(session.start())
                        elif msg_type == "session_stop":
                            await session.stop()
                            session.send_json({"type": "state", "value": "idle"})
                        else:
                            logger.info(f"[ws] Unknown message type: {msg_type}")

                    except json.JSONDecodeError as e:
                        logger.error(f"[ws] Failed to parse message: {e}")

            elif message["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"[ws] WebSocket error: {e}")
    finally:
        logger.info("[ws] Client disconnected")
        await session.stop()
        # Clean up short-term memory for this session
        agent_core.clear_session(session.session_id)


# ─── Startup Banner ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_banner():
    init_db()
    init_telemetry()

    stats = agent_core.get_stats()
    ltm_stats = stats["long_term"]
    logger.info(f"""
  ╔══════════════════════════════════════════╗
  ║     VOICE AI ASSISTANT + AGENT CORE      ║
  ║──────────────────────────────────────────║
  ║  Status:  Running                        ║
  ║  Port:    {str(PORT).ljust(33)}║
  ║  URL:     http://localhost:{str(PORT).ljust(14)}║
  ║  Model:   Amazon Nova Sonic v1           ║
  ║  Region:  {str(AWS_REGION).ljust(33)}║
  ║  Runtime: FastAPI + Uvicorn              ║
  ║──────────────────────────────────────────║
  ║  Agent Core:  ACTIVE                     ║
  ║  Short-Term:  In-Memory (TF-IDF)         ║
  ║  Long-Term:   FAISS ({str(ltm_stats['total_vectors']).ljust(4)} vectors)       ║
  ║  Documents:   {str(ltm_stats['total_documents']).ljust(4)} indexed             ║
  ║  Knowledge:   {KNOWLEDGE_DOCS_DIR.ljust(27)}║
  ╚══════════════════════════════════════════╝
    """)

    # Auto-ingest documents from knowledge_docs/ directory
    # Run in a background thread so it doesn't block Uvicorn startup (FAISS can be slow on Windows)
    async def run_auto_ingest():
        try:
            # Run the synchronous function in a thread pool
            await asyncio.to_thread(auto_ingest_documents)
        except Exception as e:
            logger.error(f"[AutoIngest] Failed during startup: {e}")
            traceback.print_exc()

    asyncio.create_task(run_auto_ingest())

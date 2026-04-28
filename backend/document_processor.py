"""
Document Processor — Ingestion Pipeline
========================================

Handles the full lifecycle of document ingestion:
  1. Upload    → Store raw file in S3 (optional backup)
  2. Extract   → Parse PDF/TXT/DOCX/CSV into raw text
  3. Chunk     → Split text into overlapping chunks (512 tokens, 50 overlap)
  4. Embed     → Generate vector embeddings via Bedrock Titan Embed v2
  5. Index     → Store vectors + metadata in FAISS (Long-Term Memory)

Supported file types:
  • PDF  — via PyPDF2
  • TXT  — direct read
  • DOCX — via python-docx
  • CSV  — via csv module (rows joined as text)

Design decisions:
  • Chunking uses character-based splitting with overlap to ensure context
    isn't lost at boundaries.
  • Each chunk gets its own embedding and is independently searchable.
  • Titan Embed v2 produces 1024-dimensional vectors.
  • S3 upload is optional — documents are always processed locally first.
"""

import io
import os
import csv
import json
import uuid
import logging
from dataclasses import dataclass
from typing import Optional

import boto3

from memory_long_term import DocumentChunk, LongTermMemory

logger = logging.getLogger("voice-server")


# ─── Configuration ───────────────────────────────────────────────────────────

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSION = 1024
CHUNK_SIZE = 1500        # Characters per chunk (≈375 tokens)
CHUNK_OVERLAP = 200      # Overlap between chunks in characters


# ─── Embedding Service ───────────────────────────────────────────────────────

class EmbeddingService:
    """
    Generates text embeddings using Amazon Bedrock Titan Embed v2.

    Each embedding is a 1024-dimensional float vector. The service handles
    batching and error recovery.

    Usage:
        service = EmbeddingService(region="us-east-1")
        vectors = await service.generate_embeddings(["Hello world", "Another text"])
        print(len(vectors[0]))  # 1024
    """

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None

    @property
    def client(self):
        """Lazy-init the Bedrock Runtime client."""
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
            )
        return self._client

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a single embedding vector for a text string.

        Args:
            text: Input text (max ~8,000 tokens for Titan Embed v2).

        Returns:
            List of 1024 floats representing the embedding.
        """
        body = json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSION,
            "normalize": True,
        })

        response = self.client.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        result = json.loads(response["body"].read())
        return result["embedding"]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Processes sequentially (Titan Embed doesn't support batching natively).
        TODO: Add asyncio.gather for parallel processing if needed.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (same order as input).
        """
        embeddings = []

        for i, text in enumerate(texts):
            try:
                # Truncate if too long (Titan has ~8000 token limit)
                truncated = text[:20000]  # ~5000 tokens, safe limit
                embedding = self.generate_embedding(truncated)
                embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.info(f"[Embed] Generated {i + 1}/{len(texts)} embeddings")

            except Exception as e:
                logger.error(f"[Embed] Error on chunk {i}: {e}")
                # Use zero vector as fallback (will have poor similarity)
                embeddings.append([0.0] * EMBEDDING_DIMENSION)

        logger.info(f"[Embed] Generated {len(embeddings)} embeddings total")
        return embeddings


# ─── Text Extraction ─────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")

        return "\n\n".join(pages)

    except ImportError:
        logger.error("[DocProcessor] PyPDF2 not installed. Cannot process PDF files.")
        raise ValueError("PDF processing not available. Install PyPDF2.")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    except ImportError:
        logger.error("[DocProcessor] python-docx not installed. Cannot process DOCX files.")
        raise ValueError("DOCX processing not available. Install python-docx.")


def extract_text_from_csv(file_bytes: bytes) -> str:
    """Extract text from a CSV file (each row becomes a paragraph)."""
    text_content = file_bytes.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text_content))

    rows = []
    headers = None

    for i, row in enumerate(reader):
        if i == 0:
            headers = row
            continue
        if headers:
            # Create key-value pairs for readability
            pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v.strip()]
            rows.append(", ".join(pairs))
        else:
            rows.append(", ".join(row))

    return "\n".join(rows)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file."""
    return file_bytes.decode("utf-8", errors="replace")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Route to the correct extractor based on file extension.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used to detect type).

    Returns:
        Extracted text as a single string.
    """
    ext = os.path.splitext(filename)[1].lower()

    extractors = {
        ".pdf": extract_text_from_pdf,
        ".txt": extract_text_from_txt,
        ".docx": extract_text_from_docx,
        ".csv": extract_text_from_csv,
        ".md": extract_text_from_txt,
        ".json": extract_text_from_txt,
    }

    if ext not in extractors:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(extractors.keys())}"
        )

    text = extractors[ext](file_bytes)

    if not text or not text.strip():
        raise ValueError(f"No text content could be extracted from '{filename}'")

    logger.info(f"[DocProcessor] Extracted {len(text)} chars from '{filename}' ({ext})")
    return text


# ─── Text Chunking ───────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks for embedding.

    The overlap ensures that context at chunk boundaries isn't lost.

    Args:
        text: Full document text.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.

    Example:
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=400, overlap=50)
        # → 3 chunks: [0:400], [350:750], [700:1000]
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for the last period, newline, or sentence-ender near the end
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.5:  # Only if it's past halfway
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward, but overlap with the previous chunk
        start = end - overlap

        # Prevent infinite loops
        if start >= len(text) - overlap:
            break

    logger.info(f"[DocProcessor] Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ─── Document Processor ─────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Orchestrates the full document ingestion pipeline.

    Usage:
        processor = DocumentProcessor(
            long_term_memory=ltm,
            embedding_service=embedding_svc,
            s3_bucket="my-bucket",
        )

        result = processor.ingest_document(file_bytes, "policy.pdf")
        print(result)
        # {
        #     "doc_id": "a1b2c3...",
        #     "filename": "policy.pdf",
        #     "chunks": 15,
        #     "characters": 22340,
        # }
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory,
        embedding_service: EmbeddingService,
        s3_bucket: Optional[str] = None,
        s3_region: str = "us-east-1",
    ):
        self.ltm = long_term_memory
        self.embedding_service = embedding_service
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region

    def ingest_document(self, file_bytes: bytes, filename: str) -> dict:
        """
        Full ingestion pipeline: Extract → Chunk → Embed → Index.

        Args:
            file_bytes: Raw file content.
            filename: Original filename.

        Returns:
            Dictionary with ingestion results.
        """
        doc_id = str(uuid.uuid4())

        logger.info(f"[DocProcessor] Starting ingestion of '{filename}' (doc_id: {doc_id[:8]}…)")

        # Step 1: Upload to S3 (optional backup)
        if self.s3_bucket:
            self._upload_to_s3(file_bytes, filename, doc_id)

        # Step 2: Extract text
        text = extract_text(file_bytes, filename)

        # Step 3: Chunk text
        chunks = chunk_text(text)

        # Step 4: Generate embeddings
        logger.info(f"[DocProcessor] Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_service.generate_embeddings(chunks)

        # Step 5: Create DocumentChunk objects with metadata
        doc_chunks = [
            DocumentChunk(
                text=chunk,
                source=filename,
                page=0,  # TODO: Track page numbers for PDFs
                chunk_index=i,
                doc_id=doc_id,
            )
            for i, chunk in enumerate(chunks)
        ]

        # Step 6: Add to Long-Term Memory (FAISS index)
        total_vectors = self.ltm.add_documents(doc_chunks, embeddings)

        result = {
            "doc_id": doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "characters": len(text),
            "total_vectors": total_vectors,
        }

        logger.info(
            f"[DocProcessor] ✅ Ingestion complete: '{filename}' → "
            f"{len(chunks)} chunks, {len(text)} chars"
        )

        return result

    def delete_document(self, doc_id: str) -> dict:
        """
        Delete a document from the index.

        Args:
            doc_id: The document's unique ID.

        Returns:
            Dictionary with deletion results.
        """
        removed = self.ltm.delete_document(doc_id)

        # Also delete from S3 if configured
        if self.s3_bucket:
            self._delete_from_s3(doc_id)

        return {
            "doc_id": doc_id,
            "chunks_removed": removed,
        }

    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        return self.ltm.list_documents()

    def _upload_to_s3(self, file_bytes: bytes, filename: str, doc_id: str) -> None:
        """Upload the raw file to S3 for backup."""
        try:
            s3 = boto3.client("s3", region_name=self.s3_region)
            key = f"documents/{doc_id}/{filename}"
            s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=file_bytes,
            )
            logger.info(f"[DocProcessor] Uploaded to S3: s3://{self.s3_bucket}/{key}")
        except Exception as e:
            logger.warning(f"[DocProcessor] S3 upload failed (non-critical): {e}")

    def _delete_from_s3(self, doc_id: str) -> None:
        """Delete the raw file from S3."""
        try:
            s3 = boto3.client("s3", region_name=self.s3_region)
            # List and delete all objects with this doc_id prefix
            response = s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f"documents/{doc_id}/",
            )
            if "Contents" in response:
                for obj in response["Contents"]:
                    s3.delete_object(Bucket=self.s3_bucket, Key=obj["Key"])
                logger.info(f"[DocProcessor] Deleted from S3: documents/{doc_id}/")
        except Exception as e:
            logger.warning(f"[DocProcessor] S3 deletion failed (non-critical): {e}")

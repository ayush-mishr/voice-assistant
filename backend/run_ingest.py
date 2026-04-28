import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["FAISS_INDEX_PATH"] = "./faiss_data"
os.environ["KNOWLEDGE_DOCS_DIR"] = "./knowledge_docs"

from document_processor import DocumentProcessor, EmbeddingService
from memory_long_term import LongTermMemory
from server import auto_ingest_documents, doc_processor, long_term_memory

print("Starting auto_ingest...")
auto_ingest_documents()
print("Done. FAISS vectors:", long_term_memory.index.ntotal if long_term_memory.index else 0)

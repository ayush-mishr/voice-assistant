# 🧠 Agent Core Technical Guide

This document is the definitive masterclass on how the **Voice Assistant Agent Core** integrates local memory (FAISS) with high-end AWS Generative AI (Titan Embed v2 & Nova Sonic), avoiding the monstrous costs of managed cloud vector databases.

If you ever need to rebuild this from scratch, just follow this step-by-step master plan.

---

## 1. The Strategy: Why Local Instead of AWS?

In a standard AWS RAG (Retrieval-Augmented Generation) pipeline, you would use **AWS Bedrock Knowledge Bases**. However, doing so requires booting an **Amazon OpenSearch Serverless** cluster to hold your data.
*   **The Trap:** OpenSearch costs between $0.24 - $0.50 per hour just to run (up to $300/month).
*   **The Solution:** We cut out OpenSearch entirely. We use AWS for its intelligence (Titan Vectors and Nova Sonic Voice), but we store the generated vectors locally in our own Python application using **FAISS** (developed by Meta). The storage cost drops instantly to **$0.00**.

Because FAISS lives inside the application folder (`backend/faiss_data`), when you containerize the app via Docker and upload it to AWS ECS, your "database" goes with you automatically. No network latency, no IAM VPC configuration, no massive bills.

---

## 2. AWS Setup (Authentication & IAM)

Before writing code, AWS needs to know who exactly is knocking on its door. 

### Local Development (Using Keys)
For testing on your laptop, we used standard AWS Access Keys. 
1. Go to AWS IAM -> Users -> Create Access Key.
2. Put the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in your `.env` file.
3. The Python library `boto3` sees these keys automatically and opens the encrypted bridge.

### Production Environment (Keyless Entry)
When deploying to AWS ECS, hardcoding API keys is dangerous.
1. We deleted the `.env` keys.
2. We gave our Docker container an **ECS Task Role** (a digital VIP badge) inside `ecs-backend-task.json`.
3. When `boto3` triggers inside the cloud, AWS detects the badge and silently passes temporary, expiring credentials to the code under the hood. Absolutely no keys exist directly in the production environment.

### Model Access
By default, AWS locks down Bedrock models. 
1. Log into AWS Console -> Bedrock -> Model Access.
2. Manually tick the checkboxes to gain access to **Amazon Nova Sonic** and **Titan Text Embeddings V2**.

---

## 3. Step-by-Step Execution: How Data Becomes Memory

Here is the exact pipeline from when a file touches the system to when the voice assistant speaks.

### Step A: Auto-Ingestion (`server.py`)
> [!NOTE]
> The orchestrator.
When the backend boots, it runs a function called `auto_ingest_documents()`. It stares at the `backend/knowledge_docs/` folder. If it sees a `.txt` or `.docx` file that it hasn't mapped before, it grabs the raw file data and ships it to the Document Processor.

### Step B: The Chopper (`document_processor.py`)
> [!IMPORTANT]
> A machine learning model cannot read an entire book in one glance. It needs paragraphs.
1. **Extraction:** Python uses normal parsing rules (like PyPDF2) to pull the pure text structure out of the document.
2. **Chunking Algorithm:** The raw text is fed into `chunk_text()`. This breaks the massive string of text into `1,500 character` paragraphs. We configure it to have a `200 character overlap` so that trailing sentences aren't awkwardly chopped in half.

### Step C: AWS Titan Integration (`EmbeddingService`)
For every text block, Python opens a connection via `boto3.client("bedrock-runtime")` directly connecting to `amazon.titan-embed-text-v2:0`.

```python
# From backend/document_processor.py
body = json.dumps({
    "inputText": chunk_text_data,
    "dimensions": 1024,
    "normalize": True,
})

response = self.client.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    contentType="application/json",
    accept="application/json",
    body=body,
)
```
AWS Titan's only job is translation. It reads the English paragraph and translates the underlying meaning/concept into a list of exactly `1,024` floating numbers (a Vector array). Titan then vanishes.

### Step D: The Local Memory Vault (`memory_long_term.py`)
> [!TIP]
> This is why your server is so fast.
We hold the original English text in one hand, and the 1,024-number vector in the other. We feed both into **FAISS** (Facebook AI Similarity Search). FAISS is an ultra-fast algorithm built entirely for categorizing these numbers. The data is locked inside `faiss_data/index.bin`. The memory is solid.

---

## 4. Real-Time Conversation (The Agent Core)

So how does the voice assistant actually *know* this data when you start talking?

### Bidirectional Streaming Rules
We use Amazon Nova Sonic via a **WebSocket**. Unlike old chat-bots where you type a question, hit enter, and wait—Nova Sonic listens to you in real-time. Because of this, we cannot search the database *after* you ask the question. 

### The Master Injection (`agent_core.py`)
Before your microphone transmits a single byte of audio, `server.py` calls `AgentCore.build_session_prompt()`. 

1. `AgentCore` immediately dips into the FAISS memory.
2. It pulls *every single line of knowledge* stored inside your localized database.
3. It bundles all that knowledge straight into the **System Instructions** template.

```python
# From backend/agent_core.py
SESSION_RAG_PROMPT = (
    "You are a smart voice assistant. Use ONLY the information below to answer.\n\n"
    "=== KNOWLEDGE BASE START ===\n"
    f"{all_faiss_document_text_injected_here}\n"
    "=== KNOWLEDGE BASE END ===\n\n"
)
```

### The Awakening
That massive, dynamically-built prompt is fired over to AWS as part of the `contentStart` stream event. When Amazon Nova Sonic wakes up to listen to your voice, it is reading from that prompt. 

When you ask, *"What college did I go to?"*, the AI doesn't search the internet. It scans the hidden instructions we pasted at the start of the session, sees the embedded resume text, and speaks back to you securely and accurately.

---

## 5. Summary Check
To build this yourself:
1. Ensure `.env` or IAM Task Roles are configured.
2. Ensure you have unlocked Bedrock Models on the AWS dash.
3. Feed the raw text into Chunks.
4. Pass the Chunks to `bedrock-runtime` Titan for mathematical mapping.
5. Store the map in FAISS.
6. Extract the map and inject it into the AI Prompt *before* the voice stream connects.

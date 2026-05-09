# Voice Assistant: Architecture, Latency, and Future Updates

This document outlines the operational details, upcoming issues, and strategies for handling the voice-to-voice communication pipeline in the application.

---

## 1. Handling User Interruptions (Barge-in)

**What will happen currently without handling?**
If the chatbot is speaking and the user interrupts, both audio streams will overlap. The system will process the new audio as a completely separate turn, but the frontend will keep playing the previous audio sequence, leading to a confusing user experience.

**How to handle this (Barge-in functionality):**
To fix this, you must implement Voice Activity Detection (VAD) on the client side and a cancellation mechanism on the backend. When the user starts speaking, the frontend immediately stops current audio playback and sends an `interrupt` signal to the backend. The backend then cancels any ongoing LLM generation or Text-to-Speech (TTS) tasks.

**Proposed Functions:**

*Frontend (JavaScript - e.g., in `useAudioPlayback.js` or `useWebSocket.js`)*:
```javascript
// Function to handle when microphone detects user speaking
function handleUserInterrupt() {
    // 1. Immediately stop current audio playback
    if (audioSourceNode) {
        audioSourceNode.stop();
    }
    clearAudioQueue(); 
    setIsPlaying(false);

    // 2. Send interruption signal to the backend
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'interrupt', message: 'User initiated barge-in' }));
    }
}
```

*Backend (Python - `server.py` and `agent_core.py`)*:
```python
# In server.py's websocket loop
async def handle_websocket(websocket, path):
    current_task = None
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "interrupt":
                if current_task and not current_task.done():
                    current_task.cancel() # Stop the LLM/TTS generation
                await websocket.send(json.dumps({"type": "interrupted_acknowledged"}))
            elif data.get("type") == "audio_input":
                # Start new processing task
                current_task = asyncio.create_task(process_audio_and_respond(websocket, data))
    except websockets.exceptions.ConnectionClosed:
        pass

# In agent_core.py
async def generate_response(prompt):
    try:
        # LLM Invocation logic here
        pass
    except asyncio.CancelledError:
        print("Generation cancelled due to user interruption")
        raise
```

---

## 2. Latency between `agent_core` and Nova Sonic

**What is happening between them?**
When the user finishes speaking, exactly four major steps occur between `agent_core` and the Amazon Bedrock Nova Sonic model:

1. **Context Assembly (RAG & Memory)**: `agent_core` queries the FAISS vector database (`faiss_data/index.faiss`) to find relevant project documentation or facts. It also loads short-term history and long-term memory.
2. **Network Trip to AWS**: The constructed prompt is sent over the internet to the AWS Bedrock API (`boto3` client invoking Nova Sonic).
3. **Model Processing (TTFT)**: Nova Sonic processes the prompt. The "Time To First Token" (TTFT) dictates how long it takes for the model to "think" before outputting the first word.
4. **TTS Conversion**: Once the text response is generated, it must be sent to a Text-to-Speech service before it can be streamed to the frontend.

**Reason for Latency:**
- **Accumulated Processing Time**: Text transcription (STT) + FAISS Search + Network Latency + LLM processing + Audio generation (TTS). 
- If you are waiting for the *entire* response from Nova Sonic before starting Text-To-Speech, the latency will be highly noticeable. 

**Solution:** Use **Streaming**. Stream tokens from Nova Sonic to the backend, and the moment you have a full sentence, stream it to your TTS engine, and stream the resulting audio chunks directly back to the React frontend.

---

## 3. Enhancing Chat Response & Topic Precision

Nova Sonic is highly capable, but to keep the responses tight and precise to the relevant topic, you need to constrain its behavior.

**How to enhance precision:**

1. **Strict System Prompting**: Update your `agent_core.py` system prompt to strictly define constraints.
   *Example:* 
   > "You are a concise voice assistant. You must answer the user's query in no more than 3 sentences. Do not use filler words like 'Furthermore' or 'Additionally'. If the user's query is unrelated to the provided context, state that you do not know. Do not hallucinate."
2. **Lower Temperature**: When invoking Bedrock, set the `temperature` parameter lower (e.g., `0.2` to `0.4`). A lower temperature reduces randomness and keeps the model focused on the most probable, precise answers.
3. **Restrictive RAG Prompting**: When passing FAISS context into the prompt, explicitly instruct the model to *only* use the provided context.
   *Example:* "Using ONLY the following documentation, answer the user's question..."
4. **Conversational Pruning**: Limit the `memory_short_term` to only the last 3-4 exchanges. Too much conversational history causes the LLM to get distracted by older topics and lose focus on the immediate question.

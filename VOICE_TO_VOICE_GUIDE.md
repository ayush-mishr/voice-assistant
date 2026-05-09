# Voice-to-Voice Architecture: Interruptions, Latency & Optimization

This document outlines upcoming issues, implementation details, and optimizations for the voice-to-voice streaming chatbot.

---

## 1. Interruption Handling (Barge-in)

When a user interrupts the chatbot while it is speaking, the system must immediately halt the current audio output and cancel the ongoing generation process to listen to the new input.

### How It Works:
1. **Frontend Voice Activity Detection (VAD):** `useAudioCapture` detects the user speaking (volume passes a certain threshold).
2. **Frontend Stop:** `useAudioPlayback` is commanded to immediately clear its audio queue and stop the Web Audio API context.
3. **Backend Cancellation:** A WebSocket signal (`{ "action": "interrupt" }`) is sent to the backend. The backend (`server.py` / `agent_core.py`) cancels the current Amazon Nova Sonic generation thread and flushes its outgoing buffers.

### Required Functions & Hooks

**Frontend (`useAudioPlayback.js`):**
```javascript
const clearPlaybackQueue = useCallback(() => {
    audioQueue.current = [];
    if (sourceNodeRef.current) {
        sourceNodeRef.current.stop();
        sourceNodeRef.current.disconnect();
        sourceNodeRef.current = null;
    }
    setIsPlaying(false);
}, []);
```

**Frontend (`useAudioCapture.js` & `useWebSocket.js` coordination):**
```javascript
const onSpeechDetected = () => {
    // 1. Stop current audio
    clearPlaybackQueue();
    // 2. Notify backend to kill current generation
    sendMessage({ action: "interrupt" });
};
```

**Backend (`agent_core.py` / `server.py`):**
```python
async def handle_interrupt(self):
    """Called when an interrupt signal is received over WebSocket"""
    self.is_interrupted = True
    # Cancel the current Bedrock/Nova Sonic streaming task
    if self.current_generation_task:
        self.current_generation_task.cancel()
    # Clear any chunks waiting to be sent to the client
    self.audio_queue.empty()
```

---

## 2. Latency Pipeline (Agent Core ↔ Nova Sonic)

If the chatbot has high latency (delay between user stopping speech and bot starting to speak), it is usually due to the pipeline between **Agent Core** and **Amazon Nova Sonic**.

### What Causes the Latency?
1. **VAD Silence Detection Delay (~500ms - 800ms):** The frontend waits to make sure the user has actually finished speaking before sending the final slice of audio.
2. **STT (Speech-to-Text) Processing:** If audio is converted to text *before* hitting Nova Sonic, this adds processing time.
3. **Agent Core Orchestration (RAG/Memory):** Before calling Nova Sonic, `agent_core.py` fetches conversation history from memory and runs embeddings to search the FAISS vector database.
4. **Nova Sonic TTFT (Time To First Token/Audio):** Amazon Bedrock needs to ingest the system prompt, history, and user input, then start generating the raw audio stream.
5. **Chunking & Buffering:** The backend waits for a complete audio chunk (e.g., 4KB) before sending it over the WebSocket to prevent choppy audio.

### How to Optimize:
* **Reduce VAD Silence Threshold:** Lower the silence timeout in `useAudioCapture` so it cuts off faster.
* **Pre-warm Connections:** Keep the Bedrock AWS client session alive.
* **Stream Audio Directly:** Ensure you are using Nova Sonic's native streaming (`ConverseStream` API) and mapping the audio bytes directly to the WebSocket without waiting for the full response to finish.

---

## 3. Enhancing Response Precision

To keep responses accurate and concise, you must strictly manage the Agent's context and LLM parameters.

### Strategies:
1. **System Prompt Tuning:** Give strict boundaries in `agent_core.py`.
   ```python
   system_prompt = """
   You are an AI assistant.
   Rules:
   1. Keep answers under 2 sentences.
   2. Do not use filler words (e.g., "Sure", "I can help with that").
   3. Only answer based on the provided retrieved context. Include nothing else.
   """
   ```
2. **Top-K RAG Optimization:** When querying the FAISS index, only return the top 2-3 most relevant chunks instead of dumping a huge document into the prompt.
3. **Generation Parameters:** Lower the `temperature` to reduce hallucinations.
   ```python
   inferenceConfig={
       "temperature": 0.2, # Lower = more focused/precise, Higher = more creative
       "maxTokens": 150    # Force short answers
   }
   ```

---

## 4. Required Hooks Summary

The React architecture for real-time voice requires three specific custom hooks:

1. **`useAudioCapture.js`**
   * Handles accessing the user's microphone (`navigator.mediaDevices.getUserMedia`).
   * Implements Voice Activity Detection (VAD) via an AnalyserNode to detect volume spikes.
   * Chunks audio into Base64 or Int16 PCM and triggers the socket send.
2. **`useWebSocket.js`**
   * Maintains the persistent WSS connection to the backend.
   * Sends audio chunks `[{ "action": "audio", "data": "..." }]`.
   * Sends interrupts `[{ "action": "interrupt" }]`.
   * Receives incoming audio chunks and passes them to the playback hook.
3. **`useAudioPlayback.js`**
   * Maintains a queue of incoming audio chunks.
   * Uses `AudioContext` to decode and seamlessly schedule audio chunks for playback.
   * Exposes a `clearPlaybackQueue()` method to handle user barge-in/interruptions.

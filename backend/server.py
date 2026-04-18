"""
Voice AI Assistant — FastAPI WebSocket Relay Server
Bridges browser WebSocket ↔ Amazon Nova Sonic bidirectional streaming.

Mirrors the logic from the original Node.js server.js exactly:
  - /ws WebSocket endpoint
  - JSON control messages: session_start, session_stop, state, transcript, error
  - Binary audio frames: PCM audio in/out
  - Bedrock event sequence: sessionStart → promptStart → system prompt → audio contentStart
    → audioInput… → contentEnd → promptEnd → sessionEnd
"""

import os
import json
import uuid
import base64
import asyncio
import logging
import traceback
import boto3

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from auth import auth_router, init_db

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

# ─── Resolve AWS Credentials (works locally via .env AND in ECS via Task Role)
# boto3 auto-discovers creds from env vars, ~/.aws, ECS metadata, etc.
# We then inject them into os.environ so EnvironmentCredentialsResolver can read them.
_boto_session = boto3.Session(region_name=os.getenv("AWS_REGION", "us-east-1"))
_boto_creds = _boto_session.get_credentials()
if _boto_creds:
    _frozen = _boto_creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = _frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = _frozen.secret_key
    if _frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = _frozen.token

# ─── Configuration ────────────────────────────────────────────────────────────

PORT = int(os.getenv("PORT", "3000"))
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "amazon.nova-sonic-v1:0"

SYSTEM_PROMPT = (
    "You are a smart voice assistant capable of answering general questions, "
    "controlling smart home devices, assisting with PC and application control, "
    "and handling specialized domain queries. Be concise, clear, and conversational. "
    "Respond as if speaking out loud. Avoid lists, bullet points, or markdown formatting "
    "in your responses. Keep your responses short, generally two or three sentences."
)

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


# ─── Bedrock Client (module-level singleton) ─────────────────────────────────

def create_bedrock_client() -> BedrockRuntimeClient:
    """Create a Bedrock Runtime client.
    
    Credentials are resolved by EnvironmentCredentialsResolver which reads
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN from env vars.
    These are set at module load time by boto3 (which auto-discovers from .env
    locally or ECS Task Role in production).
    """
    config = Config(
        endpoint_uri=f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com",
        region=AWS_REGION,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    return BedrockRuntimeClient(config=config)


# ─── Per-Connection Session Manager ──────────────────────────────────────────

class BedrockSession:
    """Manages a single bidirectional stream session with Bedrock for one WebSocket client."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.is_active = False
        self.stream_response = None
        self.response_task: asyncio.Task | None = None
        self.sender_task: asyncio.Task | None = None
        self.prompt_name = ""
        self.audio_content_name = ""
        self.audio_buffer = bytearray()

        # Outgoing message queue — decouples Bedrock receive loop from
        # WebSocket send latency, matching Node.js fire-and-forget ws.send().
        self._send_queue: asyncio.Queue = asyncio.Queue()

    # ── Send to browser ─────────────────────────────────────────────────

    async def _sender_loop(self):
        """Dedicated coroutine that drains the send queue and writes to the
        WebSocket.  Runs independently so the Bedrock receive loop never
        blocks on WebSocket back-pressure.

        Key optimisation: on each iteration we drain ALL available items from
        the queue and merge consecutive audio (bytes) entries into a single
        WebSocket frame.  This mimics Node.js's behaviour where multiple
        synchronous ws.send(buffer) calls are coalesced by the kernel into
        fewer TCP segments, reducing per-frame overhead and event-loop yields.
        """
        while True:
            # Block until at least one item is available
            first = await self._send_queue.get()
            items = [first]

            # Drain everything currently queued (non-blocking)
            while not self._send_queue.empty():
                try:
                    items.append(self._send_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # Merge consecutive "bytes" items to reduce WebSocket writes
            merged = []
            for msg_type, data in items:
                if msg_type == "bytes" and merged and merged[-1][0] == "bytes":
                    merged[-1] = ("bytes", merged[-1][1] + data)
                else:
                    merged.append((msg_type, data))

            # Send the merged batch
            for msg_type, data in merged:
                try:
                    if self.ws.client_state == WebSocketState.CONNECTED:
                        if msg_type == "bytes":
                            await self.ws.send_bytes(data)
                        else:
                            await self.ws.send_text(data)
                except Exception:
                    pass

    def send_json(self, obj: dict):
        """Enqueue a JSON text message for the browser (non-blocking)."""
        try:
            self._send_queue.put_nowait(("text", json.dumps(obj)))
        except Exception:
            pass

    def send_audio(self, data: bytes):
        """Enqueue binary audio data for the browser (non-blocking)."""
        try:
            self._send_queue.put_nowait(("bytes", data))
        except Exception:
            pass

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
        system_content_name = str(uuid.uuid4())

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
                    "content": SYSTEM_PROMPT,
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
        logger.info("[bedrock] → system prompt (contentStart/textInput/contentEnd)")

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
                                
                                # Audio output — accumulate into batch
                                if "audioOutput" in event and event["audioOutput"].get("content"):
                                    audio_batch.extend(
                                        base64.b64decode(event["audioOutput"]["content"])
                                    )

                                # Text output (assistant response text)
                                if "textOutput" in event and event["textOutput"].get("content"):
                                    self.send_json({
                                        "type": "transcript",
                                        "role": "assistant",
                                        "text": event["textOutput"]["content"],
                                    })

                                # Content start — track role changes
                                if "contentStart" in event:
                                    if event["contentStart"].get("role") == "ASSISTANT":
                                        self.send_json({"type": "state", "value": "speaking"})

                                # Content end — go back to listening
                                if "contentEnd" in event:
                                    if (
                                        event["contentEnd"].get("type") != "TOOL"
                                        and event["contentEnd"].get("role") != "USER"
                                    ):
                                        self.send_json({"type": "state", "value": "listening"})

                                # Errors from the model
                                if "validationException" in event:
                                    msg = event["validationException"].get("message", "Validation error")
                                    logger.error(f"[bedrock] Validation error: {msg}")
                                    self.send_json({"type": "error", "message": msg})

                                if "modelStreamErrorException" in event:
                                    msg = event["modelStreamErrorException"].get("message", "Stream error")
                                    logger.error(f"[bedrock] Stream error: {msg}")
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
            return

        b64_audio = base64.b64encode(pcm_buffer).decode("utf-8")

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

        self.stream_response = None
        self.response_task = None
        self.sender_task = None
        logger.info("[bedrock] Session stopped")


# ─── WebSocket Endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("[ws] Client connected")

    session = BedrockSession(ws)

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


# ─── Startup Banner ──────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_banner():
    init_db()
    logger.info(f"""
  ╔══════════════════════════════════════════╗
  ║       VOICE AI ASSISTANT SERVER          ║
  ║──────────────────────────────────────────║
  ║  Status:  Running                        ║
  ║  Port:    {str(PORT).ljust(33)}║
  ║  URL:     http://localhost:{str(PORT).ljust(14)}║
  ║  Model:   Amazon Nova Sonic v1           ║
  ║  Region:  {str(AWS_REGION).ljust(33)}║
  ║  Runtime: FastAPI + Uvicorn              ║
  ╚══════════════════════════════════════════╝
    """)

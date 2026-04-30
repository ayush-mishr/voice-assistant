import { useRef, useCallback } from 'react';

/**
 * Custom hook for gapless audio playback of PCM chunks using AudioWorklet.
 *
 * Previous approach (scheduling individual AudioBufferSourceNode per chunk) is
 * inherently fragile: it depends on chunks arriving with consistent timing to
 * keep the scheduler fed.  If there is any jitter (common with Python/asyncio
 * backends), the scheduler runs dry and inserts audible gaps.
 *
 * This approach uses a PULL model instead:
 *   - A single AudioWorkletNode runs continuously at 24 kHz.
 *   - Incoming PCM chunks are converted to Float32 and pushed into the
 *     worklet's internal FIFO buffer via postMessage.
 *   - The worklet's process() callback continuously pulls samples from the
 *     buffer.  When the buffer is empty it outputs silence (zeros).
 *
 * This design is timing-insensitive: chunks can arrive at any rate and in
 * any size.  Playback is always smooth and gapless.
 *
 * @returns {{ handleAudioResponse, stopPlayback }}
 */
export default function useAudioPlayback() {
  const playbackContextRef = useRef(null);
  const workletNodeRef = useRef(null);
  const workletReadyRef = useRef(false);
  const initPromiseRef = useRef(null);
  const pendingChunksRef = useRef([]);
  const prebufferSamplesRef = useRef(0);

  // Keep a short lead buffer for responsiveness while still absorbing jitter.
  const PREBUFFER_SECONDS = 0.05;

  /**
   * Create the AudioContext + AudioWorklet processor (once per session).
   */
  const setupWorklet = useCallback(async () => {
    const ctx = new AudioContext({ sampleRate: 24000 });
    playbackContextRef.current = ctx;
    prebufferSamplesRef.current = Math.floor(ctx.sampleRate * PREBUFFER_SECONDS); // ~120ms lead buffer

    // Chrome suspends AudioContexts created outside a user-gesture callback.
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    // Inline AudioWorklet processor — pull-based FIFO playback
    const workletCode = `
      class PlaybackProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          this._chunks = [];       // queue of Float32Array
          this._current = null;    // chunk currently being read
          this._offset = 0;        // read position within _current
          this._queuedSamples = 0; // total samples buffered across queue/current
          this._isPrimed = false;  // begin playback only after small lead buffer
          this._prebufferSamples = 1200; // default ~50ms at 24kHz
          this._maxPrebufferSamples = 7200; // hard cap ~300ms at 24kHz

          this.port.onmessage = (e) => {
            if (e.data === null) {
              // Stop signal — clear all buffered audio
              this._chunks = [];
              this._current = null;
              this._offset = 0;
              this._queuedSamples = 0;
              this._isPrimed = false;
              return;
            }
            if (e.data && e.data.type === 'config') {
              if (Number.isFinite(e.data.prebufferSamples)) {
                this._prebufferSamples = Math.max(0, e.data.prebufferSamples | 0);
                this._maxPrebufferSamples = Math.max(this._prebufferSamples, this._maxPrebufferSamples);
              }
              return;
            }
            // e.data is a Float32Array transferred from main thread
            this._chunks.push(e.data);
            this._queuedSamples += e.data.length;
          };
        }

        process(inputs, outputs) {
          const out = outputs[0][0]; // mono channel
          let i = 0;

          // Short lead buffer smooths normal WS/event-loop jitter.
          if (!this._isPrimed) {
            if (this._queuedSamples < this._prebufferSamples) {
              while (i < out.length) out[i++] = 0;
              return true;
            }
            this._isPrimed = true;
          }

          while (i < out.length) {
            // Advance to next chunk if current one is exhausted
            if (!this._current || this._offset >= this._current.length) {
              if (this._chunks.length === 0) {
                // Buffer underrun — fill the rest with silence
                while (i < out.length) out[i++] = 0;
                this._current = null;
                // Adaptively increase future priming after each underrun.
                // This keeps latency low when network is stable, but stabilizes
                // playback automatically under jitter.
                this._prebufferSamples = Math.min(this._prebufferSamples + 240, this._maxPrebufferSamples);
                this._isPrimed = false;
                return true;
              }
              this._current = this._chunks.shift();
              this._offset = 0;
            }

            // Copy as many samples as possible from current chunk
            const avail = this._current.length - this._offset;
            const need  = out.length - i;
            const n     = Math.min(avail, need);

            out.set(this._current.subarray(this._offset, this._offset + n), i);
            this._offset += n;
            i += n;
            this._queuedSamples -= n;
          }

          return true; // keep processor alive
        }
      }
      registerProcessor('playback-processor', PlaybackProcessor);
    `;

    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    await ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const node = new AudioWorkletNode(ctx, 'playback-processor');
    node.port.postMessage({
      type: 'config',
      prebufferSamples: prebufferSamplesRef.current,
    });
    node.connect(ctx.destination);
    workletNodeRef.current = node;
    workletReadyRef.current = true;
  }, [PREBUFFER_SECONDS]);

  /**
   * Called for every binary audio chunk received from the WebSocket.
   * Converts Int16 PCM → Float32 and pushes into the worklet's buffer.
   */
  const handleAudioResponse = useCallback((arrayBuffer) => {
    // Ensure even byte length for Int16Array
    const byteLength = arrayBuffer.byteLength;
    const safeLength = byteLength - (byteLength % 2);
    if (safeLength === 0) return;
    const safeBuffer = byteLength === safeLength
      ? arrayBuffer
      : arrayBuffer.slice(0, safeLength);

    // Convert Int16 PCM to Float32
    const int16 = new Int16Array(safeBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    // Fast path — worklet is ready, send directly (zero-copy transfer)
    if (workletReadyRef.current && workletNodeRef.current) {
      workletNodeRef.current.port.postMessage(float32, [float32.buffer]);
      return;
    }

    // Worklet not ready yet — buffer chunks during initialization
    pendingChunksRef.current.push(float32);

    // Trigger one-time setup on first audio chunk
    if (!initPromiseRef.current) {
      initPromiseRef.current = setupWorklet().then(() => {
        // Flush all chunks that arrived while the worklet was being set up
        for (const chunk of pendingChunksRef.current) {
          workletNodeRef.current.port.postMessage(chunk, [chunk.buffer]);
        }
        pendingChunksRef.current = [];
      });
    }
  }, [setupWorklet]);

  /**
   * Immediately silence and tear down playback.
   */
  const stopPlayback = useCallback(() => {
    // Tell worklet to clear its internal buffer
    if (workletNodeRef.current) {
      try { workletNodeRef.current.port.postMessage(null); } catch { /* */ }
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    workletReadyRef.current = false;
    initPromiseRef.current = null;
    pendingChunksRef.current = [];

    if (playbackContextRef.current && playbackContextRef.current.state !== 'closed') {
      playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
  }, []);

  return {
    handleAudioResponse,
    stopPlayback,
  };
}

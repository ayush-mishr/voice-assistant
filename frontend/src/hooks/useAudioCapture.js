import { useRef, useCallback } from 'react';

/**
 * Custom hook for microphone capture using AudioWorklet.
 * Mirrors the original startMicrophone / stopMicrophone logic exactly.
 *
 * @param {function} onPCMData – called with ArrayBuffer of int16 PCM data
 * @param {function} onError   – called with error message string
 * @returns {{ startMicrophone, stopMicrophone }}
 */
export default function useAudioCapture(onPCMData, onError) {
  const audioContextRef = useRef(null);
  const micStreamRef = useRef(null);
  const workletNodeRef = useRef(null);

  // Keep callback refs stable
  const onPCMDataRef = useRef(onPCMData);
  onPCMDataRef.current = onPCMData;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const startMicrophone = useCallback(async () => {
    try {
      // Request microphone access
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      micStreamRef.current = micStream;

      // Create audio context at 16kHz for capture
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      // Register the AudioWorklet processor inline via Blob
      const workletCode = `
        class PCMProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this._bufferSize = 4096;
            this._buffer = new Float32Array(this._bufferSize);
            this._offset = 0;
          }

          process(inputs) {
            const input = inputs[0];
            if (!input || !input[0]) return true;

            const channel = input[0];

            for (let i = 0; i < channel.length; i++) {
              this._buffer[this._offset++] = channel[i];

              if (this._offset >= this._bufferSize) {
                // Convert float32 to int16 PCM
                const pcm = new Int16Array(this._bufferSize);
                for (let j = 0; j < this._bufferSize; j++) {
                  const s = Math.max(-1, Math.min(1, this._buffer[j]));
                  pcm[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                this.port.postMessage(pcm.buffer, [pcm.buffer]);
                this._buffer = new Float32Array(this._bufferSize);
                this._offset = 0;
              }
            }

            return true;
          }
        }

        registerProcessor('pcm-processor', PCMProcessor);
      `;

      const blob = new Blob([workletCode], { type: 'application/javascript' });
      const workletUrl = URL.createObjectURL(blob);

      await audioContext.audioWorklet.addModule(workletUrl);
      URL.revokeObjectURL(workletUrl);

      const source = audioContext.createMediaStreamSource(micStream);
      const audioWorkletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
      workletNodeRef.current = audioWorkletNode;

      audioWorkletNode.port.onmessage = (event) => {
        // Send PCM buffer directly to server as binary
        onPCMDataRef.current?.(event.data);
      };

      source.connect(audioWorkletNode);
      audioWorkletNode.connect(audioContext.destination);

      return true;
    } catch (error) {
      console.error('[mic] Error:', error);
      if (error.name === 'NotAllowedError') {
        onErrorRef.current?.('Microphone access denied. Please allow microphone access and try again.');
      } else if (error.name === 'NotFoundError') {
        onErrorRef.current?.('No microphone found. Please connect a microphone and try again.');
      } else {
        onErrorRef.current?.('Microphone error: ' + error.message);
      }
      return false;
    }
  }, []);

  const stopMicrophone = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop());
      micStreamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  return { startMicrophone, stopMicrophone };
}

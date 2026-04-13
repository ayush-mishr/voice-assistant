import { useRef, useCallback } from 'react';

/**
 * Custom hook for gapless audio playback of PCM chunks.
 *
 * Key design: instead of waiting for onended to schedule the next chunk
 * (which introduces micro-gaps causing crackling), every chunk is scheduled
 * immediately at a precise future time when it arrives. This gives the
 * Web Audio scheduler a continuous, gap-free timeline.
 *
 * @returns {{ handleAudioResponse, stopPlayback }}
 */
export default function useAudioPlayback() {
  const nextPlayTimeRef = useRef(0);
  const playbackContextRef = useRef(null);
  const gainNodeRef = useRef(null);

  // Small initial latency buffer (seconds) to absorb network jitter
  const SCHEDULE_AHEAD = 0.1;

  const getPlaybackContext = useCallback(() => {
    if (!playbackContextRef.current || playbackContextRef.current.state === 'closed') {
      playbackContextRef.current = new AudioContext({ sampleRate: 24000 });

      // Create a persistent GainNode so we can instantly mute on stop
      const gain = playbackContextRef.current.createGain();
      gain.connect(playbackContextRef.current.destination);
      gainNodeRef.current = gain;
    }
    return playbackContextRef.current;
  }, []);

  const handleAudioResponse = useCallback((arrayBuffer) => {
    const ctx = getPlaybackContext();

    // Convert Int16 PCM to Float32 for Web Audio API
    const int16 = new Int16Array(arrayBuffer);
    const float32 = new Float32Array(int16.length);

    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    // Create audio buffer
    const audioBuffer = ctx.createBuffer(1, float32.length, 24000);
    audioBuffer.getChannelData(0).set(float32);

    // Schedule playback — immediately compute the start time
    const now = ctx.currentTime;

    // If we've fallen behind (first chunk, or after a gap), re-anchor
    // with a small look-ahead buffer to absorb jitter
    if (nextPlayTimeRef.current < now) {
      nextPlayTimeRef.current = now + SCHEDULE_AHEAD;
    }

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(gainNodeRef.current);
    source.start(nextPlayTimeRef.current);

    // Advance the timeline by exactly this buffer's duration — gapless
    nextPlayTimeRef.current += audioBuffer.duration;
  }, [getPlaybackContext]);

  const stopPlayback = useCallback(() => {
    nextPlayTimeRef.current = 0;

    // Instantly silence via gain to avoid pops, then tear down
    if (gainNodeRef.current) {
      try {
        gainNodeRef.current.gain.setValueAtTime(0, playbackContextRef.current.currentTime);
      } catch {
        // Context may already be closed
      }
      gainNodeRef.current = null;
    }

    if (playbackContextRef.current && playbackContextRef.current.state !== 'closed') {
      playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
  }, []);

  return { handleAudioResponse, stopPlayback };
}

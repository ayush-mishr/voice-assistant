import { useState, useCallback, useEffect, useRef } from 'react';
import useWebSocket from './hooks/useWebSocket';
import useAudioCapture from './hooks/useAudioCapture';
import useAudioPlayback from './hooks/useAudioPlayback';
import Orb from './components/Orb';
import TranscriptPanel from './components/TranscriptPanel';
import ErrorBar from './components/ErrorBar';

// ─── Constants ──────────────────────────────────────────────────────────────

const STATES = {
  IDLE: 'idle',
  LISTENING: 'listening',
  PROCESSING: 'processing',
  SPEAKING: 'speaking',
};

const STATUS_LABELS = {
  [STATES.IDLE]: 'Ready',
  [STATES.LISTENING]: 'Listening',
  [STATES.PROCESSING]: 'Processing',
  [STATES.SPEAKING]: 'Speaking',
};

const BUTTON_LABELS = {
  [STATES.IDLE]: 'Start Listening',
  [STATES.LISTENING]: 'Stop',
  [STATES.PROCESSING]: 'Stop',
  [STATES.SPEAKING]: 'Stop',
};

// ─── App ────────────────────────────────────────────────────────────────────

let entryIdCounter = 0;

export default function App() {
  const [currentState, setCurrentState] = useState(STATES.IDLE);
  const [transcriptEntries, setTranscriptEntries] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [errorVisible, setErrorVisible] = useState(false);
  const errorTimeoutRef = useRef(null);

  // Track whether the last transcript entry is an ongoing assistant message
  const lastAssistantIdRef = useRef(null);

  // ── Error display ────────────────────────────────────────────────────────

  const showError = useCallback((message) => {
    setErrorMessage(message);
    setErrorVisible(true);

    if (errorTimeoutRef.current) clearTimeout(errorTimeoutRef.current);
    errorTimeoutRef.current = setTimeout(() => {
      setErrorVisible(false);
    }, 5000);
  }, []);

  // ── Audio playback ──────────────────────────────────────────────────────

  const { handleAudioResponse, stopPlayback } = useAudioPlayback();

  // ── Transcript logic (mirrors original append-to-last-assistant behavior) ──

  const addTranscriptEntry = useCallback((role, text) => {
    // For assistant role, append to last assistant entry if it exists
    if (role === 'assistant' && lastAssistantIdRef.current !== null) {
      setTranscriptEntries((prev) =>
        prev.map((entry) =>
          entry.id === lastAssistantIdRef.current
            ? { ...entry, text: entry.text + text }
            : entry
        )
      );
      return;
    }

    const id = ++entryIdCounter;

    if (role === 'assistant') {
      lastAssistantIdRef.current = id;
    } else {
      // New user entry means new turn — reset assistant accumulator
      lastAssistantIdRef.current = null;
    }

    setTranscriptEntries((prev) => [...prev, { id, role, text }]);
  }, []);

  const clearTranscript = useCallback(() => {
    setTranscriptEntries([]);
    lastAssistantIdRef.current = null;
  }, []);

  // ── WebSocket ───────────────────────────────────────────────────────────

  const { isConnected, sendJSON, sendBinary } = useWebSocket({
    onState: (value) => setCurrentState(value),
    onTranscript: (role, text) => addTranscriptEntry(role, text),
    onAudio: (data) => handleAudioResponse(data),
    onError: (message) => showError(message),
  });

  // ── Audio capture ───────────────────────────────────────────────────────

  const { startMicrophone, stopMicrophone } = useAudioCapture(
    (pcmBuffer) => sendBinary(pcmBuffer),
    (errorMsg) => showError(errorMsg)
  );

  // ── Session control ─────────────────────────────────────────────────────

  const startSession = useCallback(async () => {
    // Clear old transcript
    clearTranscript();

    // Start microphone
    const micStarted = await startMicrophone();
    if (!micStarted) {
      setCurrentState(STATES.IDLE);
      return;
    }

    // Tell server to start Bedrock session
    sendJSON({ type: 'session_start' });
    setCurrentState(STATES.LISTENING);
  }, [clearTranscript, startMicrophone, sendJSON]);

  const stopSession = useCallback(() => {
    // Tell server to stop Bedrock session
    sendJSON({ type: 'session_stop' });

    // Stop microphone
    stopMicrophone();

    // Stop playback
    stopPlayback();

    // Reset assistant accumulator
    lastAssistantIdRef.current = null;

    setCurrentState(STATES.IDLE);
  }, [sendJSON, stopMicrophone, stopPlayback]);

  const handleActionClick = useCallback(() => {
    if (currentState === STATES.IDLE) {
      startSession();
    } else {
      stopSession();
    }
  }, [currentState, startSession, stopSession]);

  // ── Keyboard shortcut (Space to toggle) ─────────────────────────────────

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        handleActionClick();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [handleActionClick]);

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <>
      {/* Connection Indicator */}
      <div
        className={`connection-dot${isConnected ? ' connected' : ''}`}
        id="connectionDot"
      ></div>

      {/* Main App */}
      <div id="app" className={`state-${currentState}`}>
        {/* Header */}
        <header className="header">
          <h1>Voice Assistant</h1>
          <div className="subtitle">amazon nova sonic</div>
        </header>

        {/* Orb Visualizer */}
        <Orb />

        {/* Status Label */}
        <div className="status-label" id="statusLabel">
          {STATUS_LABELS[currentState]}
        </div>

        {/* Transcript Panel */}
        <TranscriptPanel entries={transcriptEntries} />

        {/* Action Button */}
        <button
          className={`action-btn${currentState !== STATES.IDLE ? ' active' : ''}`}
          id="actionBtn"
          type="button"
          onClick={handleActionClick}
        >
          {BUTTON_LABELS[currentState]}
        </button>
      </div>

      {/* Error Bar */}
      <ErrorBar message={errorMessage} visible={errorVisible} />

      {/* Footer */}
      <div className="footer">v1.0.0</div>
    </>
  );
}

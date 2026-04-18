import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import useWebSocket from './hooks/useWebSocket';
import useAudioCapture from './hooks/useAudioCapture';
import useAudioPlayback from './hooks/useAudioPlayback';
import Orb from './components/Orb';
import TranscriptPanel from './components/TranscriptPanel';
import ErrorBar from './components/ErrorBar';
import AuthPage from './AuthPage';

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

// ─── Helpers ────────────────────────────────────────────────────────────────

let entryIdCounter = 0;
let historyIdCounter = 0;

const parseUsername = (token) => {
  if (!token) return 'User';
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return payload.sub || 'User';
  } catch (e) {
    return 'User';
  }
};

// ─── App ────────────────────────────────────────────────────────────────────

export default function App() {
  const [authToken, setAuthToken] = useState(localStorage.getItem('jwtToken'));
  const username = parseUsername(authToken);
  const [currentState, setCurrentState] = useState(STATES.IDLE);
  const [transcriptEntries, setTranscriptEntries] = useState([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [errorVisible, setErrorVisible] = useState(false);
  const errorTimeoutRef = useRef(null);

  // Track whether the last transcript entry is an ongoing assistant message
  const lastAssistantIdRef = useRef(null);

  // Use a ref to track live transcript entries so stopSession can read without stale closure
  const transcriptEntriesRef = useRef([]);
  useEffect(() => {
    transcriptEntriesRef.current = transcriptEntries;
  }, [transcriptEntries]);

  // ── Per-user chat history (persisted in localStorage) ────────────────────

  const historyKey = `chatHistory_${username}`;

  const loadHistory = useCallback(() => {
    try {
      const raw = localStorage.getItem(historyKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        // Restore Date objects from strings
        return parsed.map((h) => ({ ...h, timestamp: new Date(h.timestamp) }));
      }
    } catch (e) { /* ignore corrupt data */ }
    return [];
  }, [historyKey]);

  const [chatHistory, setChatHistory] = useState(() => loadHistory());

  // Reload history when the user changes (login/logout)
  useEffect(() => {
    setChatHistory(loadHistory());
  }, [username, loadHistory]);

  // Persist history whenever it changes
  useEffect(() => {
    localStorage.setItem(historyKey, JSON.stringify(chatHistory));
  }, [chatHistory, historyKey]);

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
    // Close any selected history view
    setSelectedHistoryId(null);

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

    // Save the current transcript to history (read from ref to avoid StrictMode duplication)
    const currentEntries = transcriptEntriesRef.current;
    if (currentEntries.length > 0) {
      const historyEntry = {
        id: ++historyIdCounter,
        timestamp: new Date(),
        entries: [...currentEntries],
        preview: currentEntries.find((e) => e.role === 'user')?.text
          || currentEntries.find((e) => e.role === 'assistant')?.text
          || 'Voice session',
      };
      setChatHistory((prev) => [historyEntry, ...prev]);
    }

    // Clear live transcript
    setTranscriptEntries([]);

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

  // ── Get entries to display (live transcript OR selected history) ─────────

  const selectedHistory = selectedHistoryId
    ? chatHistory.find((h) => h.id === selectedHistoryId)
    : null;

  const isSessionActive = currentState !== STATES.IDLE;

  // During active session, show live transcript. Otherwise show selected history.
  const displayEntries = isSessionActive
    ? transcriptEntries
    : (selectedHistory ? selectedHistory.entries : []);

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <AnimatePresence mode="wait">
      {!authToken ? (
        <motion.div
          key="auth"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          style={{ width: '100%', height: '100%' }}
        >
          <AuthPage onLogin={(token) => setAuthToken(token)} />
        </motion.div>
      ) : (
        <motion.div
          key="app"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 20 }}
          transition={{ duration: 0.5 }}
          id="app"
          className={`state-${currentState}`}
        >
          {/* Connection Indicator */}
          <div className={`connection-dot${isConnected ? ' connected' : ''}`} id="connectionDot"></div>

          {/* Left Sidebar — Chat History */}
          <div className="layout-sidebar">
            <div className="sidebar-header">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
              </svg>
              History
            </div>
            <div className="history-list">
              {chatHistory.length === 0 ? (
                <div className="history-empty">No conversations yet</div>
              ) : (
                chatHistory.map((item) => (
                  <div
                    key={item.id}
                    className={`history-item${selectedHistoryId === item.id ? ' active' : ''}`}
                    onClick={() => {
                      if (!isSessionActive) {
                        setSelectedHistoryId(
                          selectedHistoryId === item.id ? null : item.id
                        );
                      }
                    }}
                  >
                    <div className="history-item-preview">
                      {item.preview.length > 50
                        ? item.preview.substring(0, 50) + '...'
                        : item.preview}
                    </div>
                    <div className="history-item-time">
                      {item.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Main Controls Area */}
          <div className="layout-main">
            {/* Top Bar with Profile & Logout */}
            <div className="top-bar">
              <header className="header">
                <h1>Voice Assistant</h1>
                <div className="subtitle">amazon nova sonic</div>
              </header>
              <div className="user-profile">
                <div className="avatar" title={username}>
                  {username.charAt(0)}
                </div>
                <button
                  className="logout-btn"
                  onClick={() => {
                    localStorage.removeItem('jwtToken');
                    setAuthToken(null);
                    stopSession();
                  }}
                >
                  Sign Out
                </button>
              </div>
            </div>

            <div className="main-content-center">
              {/* Orb Visualizer */}
              <Orb />

              {/* Status Label */}
              <div className="status-label" id="statusLabel">
                {STATUS_LABELS[currentState]}
              </div>

              {/* Live Transcript (center) — shown during session or when viewing history */}
              {displayEntries.length > 0 && (
                <TranscriptPanel entries={displayEntries} />
              )}

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
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

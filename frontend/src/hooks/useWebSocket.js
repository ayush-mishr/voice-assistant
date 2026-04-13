import { useEffect, useRef, useCallback, useState } from 'react';

/**
 * Custom hook managing the WebSocket connection to the relay server.
 * Mirrors the original vanilla-JS connectWebSocket / sendJSON / sendBinary logic.
 *
 * @param {Object} handlers
 * @param {function} handlers.onState       – called with state string (idle|listening|processing|speaking)
 * @param {function} handlers.onTranscript  – called with { role, text }
 * @param {function} handlers.onAudio       – called with ArrayBuffer (binary audio chunk)
 * @param {function} handlers.onError       – called with error message string
 * @returns {{ isConnected, sendJSON, sendBinary }}
 */
export default function useWebSocket({ onState, onTranscript, onAudio, onError }) {
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const reconnectTimer = useRef(null);

  // Keep handler refs stable across renders
  const handlers = useRef({ onState, onTranscript, onAudio, onError });
  handlers.current = { onState, onTranscript, onAudio, onError };

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log('[ws] Connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // Binary audio from server
        handlers.current.onAudio?.(event.data);
      } else {
        // JSON control message
        try {
          const message = JSON.parse(event.data);
          switch (message.type) {
            case 'state':
              handlers.current.onState?.(message.value);
              break;
            case 'transcript':
              handlers.current.onTranscript?.(message.role, message.text);
              break;
            case 'error':
              handlers.current.onError?.(message.message);
              break;
            default:
              console.log('[ws] Unknown message:', message);
          }
        } catch (e) {
          console.error('[ws] Failed to parse message:', e);
        }
      }
    };

    ws.onclose = () => {
      console.log('[ws] Disconnected');
      setIsConnected(false);
      // Attempt reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = (error) => {
      console.error('[ws] Error:', error);
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect loop on unmount
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendJSON = useCallback((obj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  }, []);

  const sendBinary = useCallback((buffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(buffer);
    }
  }, []);

  return { isConnected, sendJSON, sendBinary };
}

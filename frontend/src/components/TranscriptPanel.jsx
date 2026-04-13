import { useEffect, useRef } from 'react';

/**
 * Transcript panel component.
 * Renders the transcript header, empty-state placeholder, and all entries.
 *
 * @param {{ entries: Array<{ id: number, role: string, text: string }> }} props
 */
export default function TranscriptPanel({ entries }) {
  const panelRef = useRef(null);

  // Auto-scroll to bottom when entries change
  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight;
    }
  }, [entries]);

  return (
    <div className="transcript-panel" id="transcriptPanel" ref={panelRef}>
      <div className="transcript-header">Transcript</div>
      <div id="transcriptContent">
        {entries.length === 0 ? (
          <div className="transcript-empty" id="transcriptEmpty">
            Start a session to begin speaking
          </div>
        ) : (
          entries.map((entry) => (
            <div
              key={entry.id}
              className={`transcript-entry role-${entry.role}`}
            >
              <span className="prefix">
                {entry.role === 'user' ? '>' : '\u2014'}
              </span>
              <span className="text">{entry.text}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

import { useState, useEffect } from "react";

export default function JobDescriptionUploader({ authToken }) {
  const [activeTab, setActiveTab] = useState("file");
  const [url, setUrl] = useState("");
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // { type: "success"|"error", message }
  const [jdLoaded, setJdLoaded] = useState(false);
  const [jdSource, setJdSource] = useState("");

  // Check if a JD is already loaded on mount
  useEffect(() => {
    if (!authToken) return;
    fetch("/api/job-description-status", {
      headers: { Authorization: `Bearer ${authToken}` },
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.loaded) {
          setJdLoaded(true);
          setJdSource(data.source);
        }
      })
      .catch(() => {});
  }, [authToken]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoading(true);
    setFileName(file.name);
    setStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload-job-description", {
        method: "POST",
        body: formData,
        headers: { Authorization: `Bearer ${authToken}` },
      });
      const data = await res.json();
      if (res.ok) {
        setStatus({ type: "success", message: `"${file.name}" uploaded successfully` });
        setJdLoaded(true);
        setJdSource(`file:${file.name}`);
      } else {
        setStatus({ type: "error", message: data.detail || "Upload failed" });
      }
    } catch {
      setStatus({ type: "error", message: "Network error during upload" });
    } finally {
      setLoading(false);
    }
  };

  const handleUrlFetch = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setStatus(null);

    try {
      const res = await fetch("/api/fetch-job-description-url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${authToken}`,
        },
        body: JSON.stringify({ url }),
      });
      const data = await res.json();
      if (res.ok) {
        setStatus({ type: "success", message: "Job description fetched from URL" });
        setJdLoaded(true);
        setJdSource(`url:${url}`);
      } else {
        setStatus({ type: "error", message: data.detail || "Fetch failed" });
      }
    } catch {
      setStatus({ type: "error", message: "Could not fetch URL" });
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    try {
      await fetch("/api/job-description", {
        method: "DELETE",
        headers: { Authorization: `Bearer ${authToken}` },
      });
      setJdLoaded(false);
      setJdSource("");
      setFileName("");
      setUrl("");
      setStatus(null);
    } catch {
      // ignore
    }
  };

  // Extract a short label from the source string
  const sourceLabel = jdSource.startsWith("file:")
    ? jdSource.replace("file:", "")
    : jdSource.startsWith("url:")
      ? new URL(jdSource.replace("url:", "")).hostname
      : jdSource;

  return (
    <div className="jd-uploader">
      <div className="jd-uploader-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10 9 9 9 8 9" />
        </svg>
        Job Description
      </div>
      <div className="jd-uploader-subtitle">Upload or paste a job posting URL</div>

      {/* Active JD indicator */}
      {jdLoaded && (
        <div className="jd-active-indicator">
          <div className="jd-active-dot" />
          <span className="jd-active-label" title={jdSource}>
            {sourceLabel.length > 24 ? sourceLabel.substring(0, 24) + "..." : sourceLabel}
          </span>
          <button className="jd-clear-btn" onClick={handleClear} title="Remove">
            ✕
          </button>
        </div>
      )}

      {/* Tab buttons */}
      <div className="jd-tab-buttons">
        <button
          className={`jd-tab${activeTab === "file" ? " active" : ""}`}
          onClick={() => setActiveTab("file")}
        >
          Upload
        </button>
        <button
          className={`jd-tab${activeTab === "url" ? " active" : ""}`}
          onClick={() => setActiveTab("url")}
        >
          URL
        </button>
      </div>

      {/* File upload */}
      {activeTab === "file" && (
        <div className="jd-upload-section">
          <label className="jd-file-label">
            <input
              type="file"
              accept=".pdf,.txt,.docx,.md"
              onChange={handleFileUpload}
              disabled={loading}
            />
            {loading
              ? "Uploading..."
              : fileName
                ? fileName
                : "Choose file"}
          </label>
          <span className="jd-file-hint">PDF, DOCX, TXT, MD</span>
        </div>
      )}

      {/* URL input */}
      {activeTab === "url" && (
        <div className="jd-url-section">
          <input
            type="url"
            className="jd-url-input"
            placeholder="https://..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={loading}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleUrlFetch();
            }}
          />
          <button
            className="jd-url-btn"
            onClick={handleUrlFetch}
            disabled={loading || !url.trim()}
          >
            {loading ? "..." : "Fetch"}
          </button>
        </div>
      )}

      {/* Status message */}
      {status && (
        <div className={`jd-status ${status.type}`}>{status.message}</div>
      )}
    </div>
  );
}

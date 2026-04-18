# ═══════════════════════════════════════════════════════════
# Voice Assistant — Backend (FastAPI + Uvicorn)
# ═══════════════════════════════════════════════════════════
FROM python:3.12-slim AS base

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY auth.py .

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose the backend port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/health')" || exit 1

# Start the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3000"]

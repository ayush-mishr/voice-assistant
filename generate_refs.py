import os

backend_files = ['backend/agent_core.py', 'backend/server.py']
with open('backend_reference_server.py', 'w', encoding='utf-8') as outfile:
    for fname in backend_files:
        with open(fname, 'r', encoding='utf-8') as infile:
            outfile.write(f"\n\n# --- FILE: {fname} ---\n\n")
            outfile.write(infile.read())

frontend_files = [
    'frontend/src/hooks/useAudioCapture.js',
    'frontend/src/hooks/useAudioPlayback.js',
    'frontend/src/hooks/useWebSocket.js'
]
with open('frontend_reference_hooks.js', 'w', encoding='utf-8') as outfile:
    for fname in frontend_files:
        with open(fname, 'r', encoding='utf-8') as infile:
            outfile.write(f"\n\n// --- FILE: {fname} ---\n\n")
            outfile.write(infile.read())

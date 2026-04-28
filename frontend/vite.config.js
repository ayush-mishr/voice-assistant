import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    port: 3000,
    proxy: {
      // Only proxy the /ws path to the backend — avoids intercepting Vite's own HMR WebSocket
      '/ws': {
        target: 'http://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
        rewriteWsOrigin: true,
        // Suppress noisy ECONNREFUSED errors when backend isn't up yet
        configure: (proxy) => {
          proxy.on('error', (err) => {
            if (err.code === 'ECONNREFUSED') {
              console.warn('[vite proxy] Backend not reachable yet (ECONNREFUSED) — is it running on port 8000?');
            }
          });
        },
      },
      // Proxy /api requests to the FastAPI backend
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Output built files to ../public so the Express server can serve them
    outDir: '../public',
    emptyOutDir: true,
  },
})

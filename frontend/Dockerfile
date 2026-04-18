# ═══════════════════════════════════════════════════════════
# Voice Assistant — Frontend (React → Nginx)
# Multi-stage build for minimal production image
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Build ────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (layer caching)
COPY package.json package-lock.json ./
RUN npm ci --production=false

# Copy source and build
COPY . .

# Override vite outDir to build into /app/dist (not ../public)
RUN npx vite build --outDir dist

# ── Stage 2: Serve with Nginx ────────────────────────────
FROM nginx:alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Nginx runs as root by default in alpine (needed for port 80)
CMD ["nginx", "-g", "daemon off;"]

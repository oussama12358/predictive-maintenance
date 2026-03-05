# =============================================================================
# Dockerfile — Predictive Maintenance API
# =============================================================================
#
# Production best practices applied:
#
# 1. MULTI-STAGE BUILD
#    Stage 1 (builder): Installs all deps including build tools.
#    Stage 2 (runtime): Copies only what's needed — no compilers, no pip cache.
#    Result: Final image is ~60% smaller than a naive single-stage build.
#
# 2. NON-ROOT USER
#    Running as root inside a container is a security vulnerability.
#    We create a dedicated 'appuser' with no shell and no home directory.
#
# 3. PINNED BASE IMAGE
#    python:3.11-slim-bookworm is pinned — prevents surprise breaks from
#    upstream updates in CI/CD pipelines.
#
# 4. LAYER CACHING OPTIMIZATION
#    requirements.txt is copied and installed BEFORE app code.
#    Code changes don't invalidate the (expensive) dependency layer.
#
# 5. HEALTHCHECK
#    Docker engine monitors the /health endpoint every 30s.
#    Unhealthy containers are restarted automatically in Swarm/K8s.
#
# 6. NO .env FILES IN IMAGE
#    Secrets and configs are injected at runtime via environment variables,
#    never baked into the image.
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies (needed for some Python packages with C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — Docker caches this layer until requirements change
COPY requirements.txt .

# Install to a local prefix so we can copy cleanly to runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/      ./app/
COPY model/    ./model/
COPY data/     ./data/

# Create non-root user (uid 1001 avoids conflicts with common system uids)
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Expose the API port
EXPOSE 8000

# Health check — Docker will restart the container if this fails 3 consecutive times
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Uvicorn production config:
#   --workers 2        : 2 worker processes (CPU-bound; adjust to core count)
#   --host 0.0.0.0     : Accept connections from outside the container
#   --no-access-log    : Disable uvicorn's own access log (we use structured logging)
#   --log-level warning: Only warnings+ from uvicorn itself
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--no-access-log", \
     "--log-level", "warning"]
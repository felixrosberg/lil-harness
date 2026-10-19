# The sandbox image: Python, lil-harness, and what the agent needs to check its work.
# Build from the repository root (compose.yaml does this):
#   docker build -f examples/docker/Dockerfile -t lil-harness-sandbox .
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install the harness. For your own project, this is the line that changes:
# `pip install lil-harness` plus whatever your task needs.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . pytest

COPY examples/docker/run_task.py ./run_task.py
COPY examples/sample_project ./seeds/sample_project

# Run as an ordinary user, not root, so even inside the container the agent can't
# change the system.
RUN useradd --create-home --uid 1000 agent
USER agent
WORKDIR /workspace

# Don't litter the workspace with __pycache__ folders.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/run_task.py"]

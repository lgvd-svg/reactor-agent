FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY reactor_agent ./reactor_agent
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir pytest

ENV JAX_PLATFORMS=cpu
# LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are read from the environment (see .env.example).
ENTRYPOINT ["reactor-agent"]
CMD ["--help"]

FROM python:3.11-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Kaleido v1 uses an external browser for Plotly image export.
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

ENV BROWSER_PATH=/usr/bin/chromium \
    PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=never \
    MPLBACKEND=Agg

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra leeq --no-dev --no-install-project

COPY . .
RUN uv sync --locked --extra leeq --no-dev

# Fail the build if the deployed image cannot run visual inspection.
RUN python -c "from k_agents.inspection.vlms import matplotlib_plotly_to_pil; import plotly.graph_objects as go; image = matplotlib_plotly_to_pil(go.Figure(go.Scatter(y=[1, 3, 2]))); image.load(); assert image.width > 0"

EXPOSE 8080

CMD ["sh", "-c", "exec python -m streamlit run application/leeq/leeq_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]

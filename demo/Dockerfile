FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV AGENTIC_RAG_API_URL=http://host.docker.internal:8100

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

RUN pip install --no-cache-dir "poetry==1.8.2" \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction --no-ansi

COPY --chown=appuser:appuser gradio_app.py ./gradio_app.py

USER appuser

EXPOSE 7860

CMD ["python", "gradio_app.py"]

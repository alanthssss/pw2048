FROM python:3.12-slim AS runtime

ARG GIT_SHA=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GIT_SHA=${GIT_SHA}

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src

RUN useradd --create-home --uid 10001 appuser
USER 10001

EXPOSE 8080
ENTRYPOINT ["python", "-m", "src.api_server"]

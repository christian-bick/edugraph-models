FROM ghcr.io/astral-sh/uv:python3.13-alpine

ARG APP_DIR="/usr/src/app"
ARG PORT=8080
ENV PORT=${PORT}
ARG ONTOLOGY_VERSION=0.1.0

ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Copy only requirements to cache them in docker layer
WORKDIR ${APP_DIR}
COPY uv.lock pyproject.toml .env* gcloud-credentials.json* ${APP_DIR}

# Creating folders, and files for a project:
COPY /src ${APP_DIR}/src

# Project initialization:
RUN uv sync --locked

RUN wget https://github.com/christian-bick/edugraph-ontology/releases/download/${ONTOLOGY_VERSION}/core-ontology.rdf
RUN wget https://github.com/christian-bick/edugraph-ontology/releases/download/${ONTOLOGY_VERSION}/core-ontology.ttl

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run waitress-serve --port=$PORT --call 'api:create_app'"]
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

USER vscode

# Mnemo backend — production image for Alibaba Cloud ECS
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MEMORYAGENT_STORE=/data

WORKDIR /app

# build deps for chromadb / hnswlib native wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY memoryagent/ ./memoryagent/
COPY server.py ./
COPY web/index.html ./web/index.html

# Chroma vector store + archive persist here (mount a volume)
RUN mkdir -p /data
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

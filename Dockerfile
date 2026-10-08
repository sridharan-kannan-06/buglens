FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BUGLENS_CACHE_DIR=/app/.cache

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so the first request does not have to.
# The name must match EMBED_MODEL in buglens/config.py, and the folder must be
# <BUGLENS_CACHE_DIR>/fastembed, which is where buglens/embeddings.py looks.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/.cache/fastembed')"

COPY . .

EXPOSE 8501

# Hosting platforms pass the port in $PORT; locally it defaults to 8501.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]

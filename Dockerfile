FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
# bake the embedding model into the image so the container starts without a download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
EXPOSE 8000
# GEMINI_API_KEY must be passed at run time: docker run -e GEMINI_API_KEY=... -p 8000:8000 tariffsense
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

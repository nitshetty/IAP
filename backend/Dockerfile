# Dockerfile for FastAPI backend
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements 
COPY requirements.txt ./requirements.txt

#install Python dependencies
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Download NLTK/TextBlob corpora
RUN python -m textblob.download_corpora && \
    python -c "import nltk; nltk.download('brown'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Copy backend code
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Lightweight Python base image
FROM python:3.12-slim

# Hugging Face Spaces run containers as a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install dependencies first to make good use of the Docker layer cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user . .

# Streamlit listens on 8501 (matches app_port in README.md)
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]

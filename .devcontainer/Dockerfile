FROM python:3.11-slim

# System deps for reportlab (freetype, libjpeg); build tools for pip wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py /app/app.py

# Streamlit config
ENV PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

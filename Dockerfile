# Gunakan Python 3.10 yang stabil untuk TensorFlow 2.20
FROM python:3.10-slim

# Non-interactive tzdata dll.
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# System deps yang umum diperlukan oleh TensorFlow/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Buat workdir
WORKDIR /app

# Copy requirements dulu (supaya layer cache)
COPY requirements.txt /app/

# Install Python deps
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy kode aplikasi
COPY . /app

# Environment untuk TF sesuai kode
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_UNSAFE_DESERIALIZATION=1

# Render akan set PORT, pastikan container listen ke 0.0.0.0:$PORT
ENV PORT=7860

# Gunakan gunicorn untuk production
# -w 2: dua worker
# -k gthread: worker thread (cukup ringan utk Flask + TF CPU)
# -t 120: timeout 120s
# Bind ke $PORT
CMD ["bash", "-lc", "gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:$PORT fixed_app:app"]

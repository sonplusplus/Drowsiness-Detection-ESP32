FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy đúng theo cấu trúc thực tế
COPY test.py .
COPY best_model_m1.keras .
COPY eye_model_int8.tflite .

# Chạy với TFLite INT8 mặc định
CMD ["python", "test.py", \
     "--tflite", "eye_model_int8.tflite"]
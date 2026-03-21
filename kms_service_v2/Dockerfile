FROM python:3.11-slim

# Dependencies for liboqs build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build liboqs
WORKDIR /opt
RUN git clone --depth 1 https://github.com/open-quantum-safe/liboqs \
 && cmake -S liboqs -B liboqs/build -G Ninja \
      -DOQS_BUILD_ONLY_LIB=ON \
      -DOQS_USE_OPENSSL=OFF \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build liboqs/build \
 && cmake --install liboqs/build --prefix /usr/local

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Try to install optional PQC libraries (non-blocking)
RUN pip install --no-cache-dir oqs pqcrypto || true

COPY . /app

EXPOSE 8002

ENV QKD_AVAILABLE=true

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

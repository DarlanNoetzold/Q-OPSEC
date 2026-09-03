FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 https://github.com/open-quantum-safe/liboqs \
 && cmake -S liboqs -B liboqs/build -G Ninja \
      -DOQS_BUILD_ONLY_LIB=ON \
      -DOQS_USE_OPENSSL=OFF \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build liboqs/build \
 && cmake --install liboqs/build --prefix /usr/local

RUN pip install --no-cache-dir oqs pqcrypto fastapi uvicorn[standard] cryptography redis httpx sqlalchemy psycopg2-binary qunetsim

WORKDIR /app
COPY . /app

EXPOSE 8002

ENV REDIS_URL=redis://host.docker.internal:6379
ENV QKD_AVAILABLE=true

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
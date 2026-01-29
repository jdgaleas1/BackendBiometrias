# ==============================================================================
# ETAPA 1: BUILD (compila tu proyecto + whisper.cpp)
# ==============================================================================
FROM debian:bullseye AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Compilar tu proyecto de voz ---
WORKDIR /build
COPY . .

# Compilar SIN libreria movil (BUILD_MOBILE_LIB=OFF por defecto)
# La libreria movil solo se compila manualmente con compilar_mobile_android.ps1
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_MOBILE_LIB=OFF && \
    cmake --build build --parallel $(nproc)

# --- Compilar whisper.cpp (shared libs + whisper-cli) ---
WORKDIR /whisper_build
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git whisper.cpp && \
    cd whisper.cpp && \
    cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel $(nproc)

# ==============================================================================
# ETAPA 2: RUNTIME
# ==============================================================================
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Tu binario principal + modelo ASR ---
# Los binarios están en build/voz/ porque CMakeLists.txt raíz usa add_subdirectory("voz")
COPY --from=builder /build/build/voz/voz_servidor_biometrico /app/voz_servidor_biometrico
COPY --from=builder /build/build/voz/ggml-tiny.bin /app/

# --- Whisper runtime: librerías y CLI ---
COPY --from=builder /whisper_build/whisper.cpp/build/src/libwhisper.so* /app/
COPY --from=builder /whisper_build/whisper.cpp/build/ggml/src/libggml*.so* /app/
COPY --from=builder /whisper_build/whisper.cpp/build/bin/whisper-cli /app/whisper-cli

# Crear estructura de directorios
RUN mkdir -p /app/model /app/processed_dataset_bin /app/temp_audio && \
    chmod -R 777 /app/temp_audio

# Compatibilidad de rutas para modelos biométricos
RUN mkdir -p /app/voz/apps && \
    ln -sfn /app/model /app/voz/apps/models && \
    mkdir -p /apps && \
    ln -sfn /app/model /apps/models

# Permisos ejecutables
RUN chmod +x /app/voz_servidor_biometrico /app/whisper-cli 2>/dev/null || true

# Variables de entorno para resolver .so
ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH \
    APP_DIR=/app \
    PATH="/app:${PATH}"

EXPOSE 8081

CMD ["./voz_servidor_biometrico"]

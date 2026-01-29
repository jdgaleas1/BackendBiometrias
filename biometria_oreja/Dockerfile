# =========================
# STAGE 1: BUILD
# =========================
FROM ubuntu:22.04 AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential cmake ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN rm -rf build-linux && mkdir -p build-linux && cd build-linux && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j"$(nproc)"

# =========================
# STAGE 2: RUNTIME
# =========================
FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    libstdc++6 \
    libgcc-s1 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/build-linux/servidor /app/servidor
COPY --from=build /app/build-linux/agregar_usuario /app/agregar_usuario
COPY --from=build /app/build-linux/agregar_usuario_biometria /app/agregar_usuario_biometria
COPY --from=build /app/build-linux/procesar_dataset /app/procesar_dataset
COPY --from=build /app/build-linux/entrenar_svm /app/entrenar_svm
COPY --from=build /app/build-linux/predecir /app/predecir
COPY --from=build /app/build-linux/evaluar_batch_qc /app/evaluar_batch_qc
COPY --from=build /app/build-linux/reentrenar_svm /app/reentrenar_svm
COPY --from=build /app/build-linux/procesar_dataset_secuencial /app/procesar_dataset_secuencial

# Seed inicial (opcional). En producción prefiero volumen.
COPY out /app/out

RUN mkdir -p /app/nuevo_usuario /app/out /tmp

# Usuario no-root
RUN useradd -m appuser && \
    chown -R appuser:appuser /app/out /tmp /app/nuevo_usuario /app && \
    chmod +x /app/servidor /app/agregar_usuario /app/agregar_usuario_biometria /app/procesar_dataset /app/entrenar_svm /app/predecir /app/reentrenar_svm /app/procesar_dataset_secuencial

USER appuser

EXPOSE 8085
CMD ["./servidor"]

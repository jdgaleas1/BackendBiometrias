#!/bin/bash
# Script Bash para inyectar frases dinamicas en el contenedor PostgreSQL
# Uso: ./inyectar_frases.sh

echo "========================================"
echo " Inyectando frases dinamicas en DB"
echo "========================================"
echo ""

# Verificar que el contenedor este corriendo
echo "[*] Verificando estado del contenedor biometria_db..."
if ! docker ps --filter "name=biometria_db" --format "{{.Status}}" | grep -q "Up"; then
    echo "[!] ERROR: El contenedor biometria_db no esta corriendo"
    echo "[*] Iniciando contenedores..."
    docker-compose -f docker-compose-multimodal.yml up -d db
    sleep 10
else
    echo "[OK] Contenedor biometria_db esta activo"
fi

# Copiar el archivo SQL al contenedor
echo ""
echo "[*] Copiando archivo SQL al contenedor..."
if docker cp insert_frases_dinamicas.sql biometria_db:/tmp/insert_frases_dinamicas.sql; then
    echo "[OK] Archivo copiado exitosamente"
else
    echo "[!] ERROR: No se pudo copiar el archivo"
    exit 1
fi

# Ejecutar el script SQL dentro del contenedor
echo ""
echo "[*] Ejecutando script SQL en la base de datos..."
if docker exec -i biometria_db psql -U biometria -d usuarios_db -f /tmp/insert_frases_dinamicas.sql; then
    echo "[OK] Script ejecutado correctamente"
else
    echo "[!] ERROR: Fallo la ejecucion del script"
    exit 1
fi

# Verificar cuantas frases se insertaron
echo ""
echo "[*] Verificando frases insertadas..."
count=$(docker exec -i biometria_db psql -U biometria -d usuarios_db -t -c "SELECT COUNT(*) FROM textos_dinamicos_audio;" | xargs)
echo "[OK] Total de frases en la base de datos: $count"

# Limpiar archivo temporal del contenedor
echo ""
echo "[*] Limpiando archivos temporales..."
docker exec -i biometria_db rm /tmp/insert_frases_dinamicas.sql

echo ""
echo "========================================"
echo " Proceso completado exitosamente"
echo "========================================"

#!/bin/bash
set -e

CONTAINER="biometria_db"
USER="biometria"
DB="usuarios_db"

echo "⚠ Eliminando datos de usuarios / credenciales / validaciones ..."
echo "Contenedor: $CONTAINER"
echo "Usuario BD: $USER"
echo "Base de datos: $DB"

SQL="
TRUNCATE TABLE credenciales_biometricas RESTART IDENTITY CASCADE;
TRUNCATE TABLE validaciones_biometricas RESTART IDENTITY CASCADE;
TRUNCATE TABLE textos_dinamicos_audio RESTART IDENTITY CASCADE;
TRUNCATE TABLE usuarios RESTART IDENTITY CASCADE;
"

docker exec -i $CONTAINER psql -v ON_ERROR_STOP=1 -U $USER -d $DB <<EOF
$SQL
EOF

echo "✔ TABLAS LIMPIADAS EXITOSAMENTE."

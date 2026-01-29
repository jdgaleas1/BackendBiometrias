#ifndef SYNC_CONTROLLER_H
#define SYNC_CONTROLLER_H

#include <string>
#include <memory>
#include "../service/sincronizacion_service.h"
#include "../../external/json.hpp"

using json = nlohmann::json;

class SyncController {
private:
    std::unique_ptr<SincronizacionService> syncService;

public:
    SyncController();
    ~SyncController();

    // ========================================================================
    // POST /sync/push
    // Mobile envia vectores de caracteristicas pendientes
    // ========================================================================
    // Body: {
    //   "uuid_dispositivo": "abc-123",
    //   "caracteristicas": [
    //     {
    //       "id_usuario": 1,
    //       "id_credencial": 5,
    //       "vector_features": [0.1, 0.2, ...],
    //       "dimension": 39
    //     },
    //     ...
    //   ]
    // }
    // Response: {"ok": true, "ids_procesados": [1, 2, 3], "procesados": 3, "total": 3}
    json syncPush(const json& body);

    // ========================================================================
    // GET /sync/pull?desde=<timestamp>
    // Mobile pide cambios del servidor desde un timestamp
    // ========================================================================
    // Query params: desde=2026-01-20T10:30:00Z (opcional)
    // Response: {
    //   "ok": true,
    //   "frases": [{id_frase, frase, updated_at}, ...],
    //   "usuarios": [{id_usuario, identificador_unico, estado, updated_at}, ...],
    //   "timestamp_actual": "2026-01-24T12:00:00Z"
    // }
    json syncPull(const std::string& desde = "");

    // ========================================================================
    // GET /sync/modelo?cedula=<cedula>
    // Mobile pide modelo re-entrenado del servidor
    // ========================================================================
    // Query params: cedula=1234567890
    // Response: binario del modelo SVM serializado
    std::vector<uint8_t> syncModelo(const std::string& cedula);
};

#endif // SYNC_CONTROLLER_H

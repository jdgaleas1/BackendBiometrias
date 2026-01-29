#include "sync_controller.h"
#include <iostream>

SyncController::SyncController() {
    syncService = std::make_unique<SincronizacionService>();
    std::cout << "@ SyncController inicializado" << std::endl;
}

SyncController::~SyncController() {
    std::cout << "@ SyncController destruido" << std::endl;
}

// ============================================================================
// POST /sync/push
// ============================================================================

json SyncController::syncPush(const json& body) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  SYNC PUSH (Mobile -> Server)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    try {
        // Validar body
        if (!body.contains("uuid_dispositivo") || !body.contains("caracteristicas")) {
            json error;
            error["ok"] = false;
            error["error"] = "Faltan campos requeridos: uuid_dispositivo, caracteristicas";
            return error;
        }

        std::string uuidDispositivo = body["uuid_dispositivo"];
        json caracteristicas = body["caracteristicas"];

        if (!caracteristicas.is_array() || caracteristicas.empty()) {
            json error;
            error["ok"] = false;
            error["error"] = "caracteristicas debe ser un array no vacio";
            return error;
        }

        std::cout << "-> UUID Dispositivo: " << uuidDispositivo << std::endl;
        std::cout << "-> Items a sincronizar: " << caracteristicas.size() << std::endl;

        // Delegar al servicio
        auto resultado = syncService->recibirCaracteristicas(caracteristicas, uuidDispositivo);
        
        std::cout << "\n@ Sync push completado" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        return resultado;

    } catch (const std::exception& e) {
        std::cerr << "! Error en sync push: " << e.what() << std::endl;
        json error;
        error["ok"] = false;
        error["error"] = e.what();
        return error;
    }
}

// ============================================================================
// GET /sync/pull
// ============================================================================

json SyncController::syncPull(const std::string& desde) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  SYNC PULL (Server -> Mobile)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    try {
        std::cout << "-> Timestamp desde: " << (desde.empty() ? "inicio" : desde) << std::endl;

        // Delegar al servicio
        auto resultado = syncService->obtenerCambiosDesde(desde);
        
        std::cout << "\n@ Sync pull completado" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        return resultado;

    } catch (const std::exception& e) {
        std::cerr << "! Error en sync pull: " << e.what() << std::endl;
        json error;
        error["ok"] = false;
        error["error"] = e.what();
        return error;
    }
}

// ============================================================================
// GET /sync/modelo
// ============================================================================

std::vector<uint8_t> SyncController::syncModelo(const std::string& cedula) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  SYNC MODELO (Server -> Mobile)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::vector<uint8_t> modeloVacio;
    
    try {
        if (cedula.empty()) {
            std::cerr << "! Cedula no proporcionada" << std::endl;
            return modeloVacio;
        }

        std::cout << "-> Cedula solicitada: " << cedula << std::endl;

        // Delegar al servicio
        auto modelo = syncService->obtenerModeloActualizado(cedula);
        
        if (modelo.empty()) {
            std::cout << "# No se pudo generar modelo para: " << cedula << std::endl;
        } else {
            std::cout << "-> Modelo generado: " << modelo.size() << " bytes" << std::endl;
        }
        
        std::cout << "\n@ Sync modelo completado" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        return modelo;

    } catch (const std::exception& e) {
        std::cerr << "! Error en sync modelo: " << e.what() << std::endl;
        return modeloVacio;
    }
}

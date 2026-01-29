#ifndef SINCRONIZACION_SERVICE_H
#define SINCRONIZACION_SERVICE_H

#include <string>
#include <vector>
#include "../../external/json.hpp"

using json = nlohmann::json;

class SincronizacionService {
public:
    SincronizacionService();
    
    // ========================================================================
    // Mobile envia sus vectores pendientes
    // ========================================================================
    // Recibe JSON con array de {id_usuario, id_credencial, vector_features[], dimension, uuid_dispositivo}
    // Inserta en caracteristicas_hablantes (PostgreSQL)
    // Retorna {ok, ids_procesados[]}
    json recibirCaracteristicas(const json& items, const std::string& uuidDispositivo);
    
    // ========================================================================
    // Mobile pide cambios desde timestamp
    // ========================================================================
    // Consulta tablas con updated_at > desde
    // Retorna {usuarios[], frases[], timestamp_actual}
    json obtenerCambiosDesde(const std::string& desde);
    
    // ========================================================================
    // Mobile pide modelo actualizado para un usuario
    // ========================================================================
    // Server re-entrena SVM con TODOS los vectores de ese usuario
    // Retorna modelo binario serializado
    std::vector<uint8_t> obtenerModeloActualizado(const std::string& cedula);
    
private:
    // Convertir vector<double> a BYTEA para PostgreSQL
    std::string vectorToByteArray(const std::vector<double>& vec);
    
    // Convertir BYTEA de PostgreSQL a vector<double>
    std::vector<double> byteArrayToVector(const std::string& byteArray);
};

#endif // SINCRONIZACION_SERVICE_H

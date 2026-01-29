#include "sincronizacion_service.h"
#include "../../utils/http_helpers.h"
#include "../../external/httplib.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>

using namespace HttpHelpers;

SincronizacionService::SincronizacionService() {
    std::cout << "@ SincronizacionService inicializado" << std::endl;
}

// ============================================================================
// RECIBIR CARACTERISTICAS DEL MOBILE
// ============================================================================

json SincronizacionService::recibirCaracteristicas(const json& items, 
                                                    const std::string& uuidDispositivo) {
    std::cout << "-> Recibiendo " << items.size() << " caracteristicas del dispositivo: " 
              << uuidDispositivo << std::endl;
    
    json response;
    response["ok"] = true;
    response["ids_procesados"] = json::array();
    
    try {
        int procesados = 0;
        
        for (const auto& item : items) {
            try {
                // Validar campos requeridos
                if (!item.contains("id_usuario") || !item.contains("vector_features") || 
                    !item.contains("dimension")) {
                    std::cerr << "# Item invalido, faltan campos requeridos" << std::endl;
                    continue;
                }
                
                int idUsuario = item["id_usuario"];
                int idCredencial = item.value("id_credencial", 0);
                std::vector<double> features = item["vector_features"];
                int dimension = item["dimension"];
                
                // Convertir vector a BYTEA
                std::string vectorBytes = vectorToByteArray(features);
                
                // Construir JSON para insercion
                json caracteristica;
                caracteristica["id_usuario"] = idUsuario;
                if (idCredencial > 0) {
                    caracteristica["id_credencial"] = idCredencial;
                }
                caracteristica["vector_features"] = vectorBytes;
                caracteristica["dimension"] = dimension;
                caracteristica["origen"] = "mobile";
                caracteristica["uuid_dispositivo"] = uuidDispositivo;
                
                // Insertar en PostgreSQL via PostgREST
                auto res = hacerPOST("/caracteristicas_hablantes", caracteristica);
                
                if (res && res->status == 201) {
                    // PostgREST retorna el registro insertado con Prefer=return=representation
                    json resultado = json::parse(res->body);
                    if (resultado.is_array() && !resultado.empty()) {
                        int idCaracteristica = resultado[0]["id_caracteristica"];
                        response["ids_procesados"].push_back(idCaracteristica);
                        procesados++;
                        std::cout << "   @ Caracteristica insertada: ID=" << idCaracteristica 
                                  << " Usuario=" << idUsuario << std::endl;
                    }
                } else {
                    std::cerr << "# Error insertando caracteristica: " 
                              << (res ? std::to_string(res->status) : "sin respuesta") << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "# Error procesando item: " << e.what() << std::endl;
            }
        }
        
        std::cout << "-> Procesadas " << procesados << "/" << items.size() 
                  << " caracteristicas correctamente" << std::endl;
        
        response["procesados"] = procesados;
        response["total"] = items.size();
        
    } catch (const std::exception& e) {
        std::cerr << "! Error recibiendo caracteristicas: " << e.what() << std::endl;
        response["ok"] = false;
        response["error"] = e.what();
    }
    
    return response;
}

// ============================================================================
// OBTENER CAMBIOS DESDE TIMESTAMP
// ============================================================================

json SincronizacionService::obtenerCambiosDesde(const std::string& desde) {
    std::cout << "-> Obteniendo cambios desde: " << (desde.empty() ? "inicio" : desde) << std::endl;
    
    json response;
    response["ok"] = true;
    response["frases"] = json::array();
    response["usuarios"] = json::array();
    
    try {
        // Obtener frases actualizadas
        std::string endpointFrases = "/textos_dinamicos_audio";
        if (!desde.empty()) {
            endpointFrases += "?updated_at=gt." + desde;
        }
        
        auto resFrases = hacerGET(endpointFrases);
        
        if (resFrases && resFrases->status == 200) {
            json frases = json::parse(resFrases->body);
            
            for (const auto& frase : frases) {
                if (frase.value("estado_texto", "") == "activo") {
                    json fraseData;
                    fraseData["id_frase"] = frase["id_texto"];
                    fraseData["frase"] = frase["frase"];
                    fraseData["updated_at"] = frase.value("updated_at", "");
                    response["frases"].push_back(fraseData);
                }
            }
        }
        
        // Obtener usuarios actualizados
        std::string endpointUsuarios = "/usuarios";
        if (!desde.empty()) {
            endpointUsuarios += "?updated_at=gt." + desde;
        }
        
        auto resUsuarios = hacerGET(endpointUsuarios);
        
        if (resUsuarios && resUsuarios->status == 200) {
            json usuarios = json::parse(resUsuarios->body);
            
            for (const auto& usuario : usuarios) {
                json userData;
                userData["id_usuario"] = usuario["id_usuario"];
                userData["identificador_unico"] = usuario["identificador_unico"];
                userData["estado"] = usuario.value("estado", "activo");
                userData["updated_at"] = usuario.value("updated_at", "");
                response["usuarios"].push_back(userData);
            }
        }
        
        // Timestamp actual
        auto now = std::time(nullptr);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
        response["timestamp_actual"] = ss.str();
        
        std::cout << "-> Cambios encontrados: " << response["frases"].size() 
                  << " frases, " << response["usuarios"].size() << " usuarios" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "! Error obteniendo cambios: " << e.what() << std::endl;
        response["ok"] = false;
        response["error"] = e.what();
    }
    
    return response;
}

// ============================================================================
// OBTENER MODELO ACTUALIZADO
// ============================================================================

std::vector<uint8_t> SincronizacionService::obtenerModeloActualizado(const std::string& cedula) {
    std::cout << "-> Generando modelo actualizado para: " << cedula << std::endl;
    
    std::vector<uint8_t> modeloVacio;
    
    try {
        // 1. Obtener ID de usuario por cedula
        std::string endpointUsuario = "/usuarios?identificador_unico=eq." + cedula;
        auto resUsuario = hacerGET(endpointUsuario);
        
        if (!resUsuario || resUsuario->status != 200) {
            std::cerr << "# Error obteniendo usuario: " << cedula << std::endl;
            return modeloVacio;
        }
        
        json usuarios = json::parse(resUsuario->body);
        if (usuarios.empty()) {
            std::cerr << "# Usuario no encontrado: " << cedula << std::endl;
            return modeloVacio;
        }
        
        int idUsuario = usuarios[0]["id_usuario"];
        std::cout << "   @ Usuario encontrado: ID=" << idUsuario << std::endl;
        
        // 2. Obtener TODAS las caracteristicas del usuario
        std::string endpointCaracteristicas = "/caracteristicas_hablantes?id_usuario=eq." 
                                               + std::to_string(idUsuario);
        auto resCaracteristicas = hacerGET(endpointCaracteristicas);
        
        if (!resCaracteristicas || resCaracteristicas->status != 200) {
            std::cerr << "# Error obteniendo caracteristicas" << std::endl;
            return modeloVacio;
        }
        
        json caracteristicas = json::parse(resCaracteristicas->body);
        
        if (caracteristicas.empty()) {
            std::cerr << "# No hay caracteristicas para reentrenar modelo" << std::endl;
            return modeloVacio;
        }
        
        std::cout << "   @ Caracteristicas encontradas: " << caracteristicas.size() << std::endl;
        
        // 3. TODO: Re-entrenar SVM con TODAS las caracteristicas
        // Por ahora retornar modelo vacio
        // En una implementacion real:
        // - Convertir BYTEA a vectores
        // - Entrenar SVM One-vs-All
        // - Serializar modelo a binario
        // - Retornar bytes del modelo
        
        std::cout << "# Reentrenamiento de modelo pendiente de implementar" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "! Error generando modelo: " << e.what() << std::endl;
    }
    
    return modeloVacio;
}

// ============================================================================
// UTILIDADES
// ============================================================================

std::string SincronizacionService::vectorToByteArray(const std::vector<double>& vec) {
    // Convertir vector<double> a string hexadecimal para BYTEA de PostgreSQL
    std::stringstream ss;
    ss << "\\x";
    
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(vec.data());
    size_t numBytes = vec.size() * sizeof(double);
    
    for (size_t i = 0; i < numBytes; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') 
           << static_cast<int>(bytes[i]);
    }
    
    return ss.str();
}

std::vector<double> SincronizacionService::byteArrayToVector(const std::string& byteArray) {
    // Convertir string hexadecimal de BYTEA a vector<double>
    // Formato esperado: \xAABBCC...
    
    if (byteArray.size() < 2 || byteArray.substr(0, 2) != "\\x") {
        throw std::runtime_error("Formato BYTEA invalido");
    }
    
    std::string hexData = byteArray.substr(2); // Remover \x
    
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hexData.length(); i += 2) {
        std::string byteStr = hexData.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoi(byteStr, nullptr, 16));
        bytes.push_back(byte);
    }
    
    // Convertir bytes a doubles
    size_t numDoubles = bytes.size() / sizeof(double);
    std::vector<double> result(numDoubles);
    
    std::memcpy(result.data(), bytes.data(), bytes.size());
    
    return result;
}

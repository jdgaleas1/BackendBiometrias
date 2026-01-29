#ifndef HTTP_HELPERS_H
#define HTTP_HELPERS_H

#include <string>
#include <thread>
#include <chrono>
#include "../external/httplib.h"
#include "../external/json.hpp"
#include "config.h"

// =============================================================================
// HELPERS HTTP PARA POSTGREST - EVITAR DUPLICACION DE CODIGO
// =============================================================================

namespace HttpHelpers {

// Crear cliente HTTP configurado correctamente para PostgREST
inline httplib::Client crearClientePostgREST(int timeoutSegundos = 15) {
    auto [host, port] = obtenerPostgRESTConfig();
    httplib::Client cli(host.c_str(), port);
    
    // CRITICO: keep_alive en false para evitar problemas en Docker
    cli.set_keep_alive(false);
    cli.set_connection_timeout(timeoutSegundos, 0);
    cli.set_read_timeout(timeoutSegundos, 0);
    cli.set_write_timeout(timeoutSegundos, 0);
    
    return cli;
}

// Headers estandar para requests a PostgREST
inline httplib::Headers headersGET() {
    return {
        {"Accept", "application/json"}
    };
}

inline httplib::Headers headersPOST() {
    return {
        {"Content-Type", "application/json"},
        {"Prefer", "return=representation"}  // CRITICO para PostgREST
    };
}

inline httplib::Headers headersPATCH() {
    return {
        {"Content-Type", "application/json"},
        {"Prefer", "return=minimal"}  // PATCH devuelve 204 sin body
    };
}

inline httplib::Headers headersDELETE() {
    return {
        {"Prefer", "return=minimal"}
    };
}

// Wrapper para GET con manejo de errores y RETRY
inline httplib::Result hacerGET(const std::string& endpoint, int timeoutSegundos = 15) {
    auto [host, port] = obtenerPostgRESTConfig();
    std::cout << "[HTTP DEBUG] GET " << endpoint << " -> " << host << ":" << port << std::endl;
    
    auto headers = headersGET();
    
    // Retry logic: 3 intentos con delay incremental
    const int MAX_RETRIES = 3;
    httplib::Result res;
    
    for (int intento = 1; intento <= MAX_RETRIES; ++intento) {
        httplib::Client cli(host.c_str(), port);
        
        // CRITICO: keep_alive=false y write_timeout para Docker
        cli.set_keep_alive(false);
        cli.set_connection_timeout(timeoutSegundos, 0);
        cli.set_read_timeout(timeoutSegundos, 0);
        cli.set_write_timeout(timeoutSegundos, 0);
        
        res = cli.Get(endpoint.c_str(), headers);
        
        if (res) {
            if (intento > 1) {
                std::cout << "[HTTP DEBUG] GET exitoso en intento " << intento << std::endl;
            }
            return res;  // Exito
        }
        
        // Fallo - logging y retry
        std::cerr << "[HTTP DEBUG] GET intento " << intento << "/" << MAX_RETRIES 
                  << " fallo - Error: " << httplib::to_string(res.error()) << std::endl;
        
        if (intento < MAX_RETRIES) {
            int delayMs = intento * 500;  // 500ms, 1000ms
            std::cout << "[HTTP DEBUG] Reintentando en " << delayMs << "ms..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
    }
    
    std::cerr << "[HTTP ERROR] GET fallo despues de " << MAX_RETRIES << " intentos" << std::endl;
    return res;  // Retornar ultimo resultado (nullptr)
}

// Wrapper para POST con manejo de errores y RETRY
inline httplib::Result hacerPOST(const std::string& endpoint, 
                                  const nlohmann::json& body,
                                  int timeoutSegundos = 15) {
    auto [host, port] = obtenerPostgRESTConfig();
    std::cout << "[HTTP DEBUG] POST " << endpoint << " -> " << host << ":" << port << std::endl;
    
    auto headers = headersPOST();
    std::string bodyStr = body.dump();
    std::cout << "[HTTP DEBUG] Body: " << bodyStr.substr(0, 200) << (bodyStr.length() > 200 ? "..." : "") << std::endl;
    
    // Retry logic: 3 intentos con delay incremental
    const int MAX_RETRIES = 3;
    httplib::Result res;
    
    for (int intento = 1; intento <= MAX_RETRIES; ++intento) {
        httplib::Client cli(host.c_str(), port);
        
        // CRITICO: keep_alive=false y write_timeout para Docker
        cli.set_keep_alive(false);
        cli.set_connection_timeout(timeoutSegundos, 0);
        cli.set_read_timeout(timeoutSegundos, 0);
        cli.set_write_timeout(timeoutSegundos, 0);
        
        res = cli.Post(endpoint.c_str(), headers, bodyStr, "application/json");
        
        if (res) {
            if (intento > 1) {
                std::cout << "[HTTP DEBUG] POST exitoso en intento " << intento << std::endl;
            }
            return res;  // Exito
        }
        
        // Fallo - logging y retry
        std::cerr << "[HTTP DEBUG] POST intento " << intento << "/" << MAX_RETRIES 
                  << " fallo - Error: " << httplib::to_string(res.error()) << std::endl;
        
        if (intento < MAX_RETRIES) {
            int delayMs = intento * 500;  // 500ms, 1000ms
            std::cout << "[HTTP DEBUG] Reintentando en " << delayMs << "ms..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
    }
    
    std::cerr << "[HTTP ERROR] POST fallo despues de " << MAX_RETRIES << " intentos" << std::endl;
    return res;  // Retornar ultimo resultado (nullptr)
}

// Wrapper para PATCH con manejo de errores y RETRY
inline httplib::Result hacerPATCH(const std::string& endpoint,
                                   const nlohmann::json& body,
                                   int timeoutSegundos = 15) {
    auto [host, port] = obtenerPostgRESTConfig();
    std::cout << "[HTTP DEBUG] PATCH " << endpoint << " -> " << host << ":" << port << std::endl;
    
    auto headers = headersPATCH();
    std::string bodyStr = body.dump();
    
    // Retry logic: 3 intentos con delay incremental
    const int MAX_RETRIES = 3;
    httplib::Result res;
    
    for (int intento = 1; intento <= MAX_RETRIES; ++intento) {
        httplib::Client cli(host.c_str(), port);
        
        // CRITICO: keep_alive=false y write_timeout para Docker
        cli.set_keep_alive(false);
        cli.set_connection_timeout(timeoutSegundos, 0);
        cli.set_read_timeout(timeoutSegundos, 0);
        cli.set_write_timeout(timeoutSegundos, 0);
        
        res = cli.Patch(endpoint.c_str(), headers, bodyStr, "application/json");
        
        if (res) {
            if (intento > 1) {
                std::cout << "[HTTP DEBUG] PATCH exitoso en intento " << intento << std::endl;
            }
            return res;  // Exito
        }
        
        // Fallo - logging y retry
        std::cerr << "[HTTP DEBUG] PATCH intento " << intento << "/" << MAX_RETRIES 
                  << " fallo - Error: " << httplib::to_string(res.error()) << std::endl;
        
        if (intento < MAX_RETRIES) {
            int delayMs = intento * 500;  // 500ms, 1000ms
            std::cout << "[HTTP DEBUG] Reintentando en " << delayMs << "ms..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
    }
    
    std::cerr << "[HTTP ERROR] PATCH fallo despues de " << MAX_RETRIES << " intentos" << std::endl;
    return res;  // Retornar ultimo resultado (nullptr)
}

// Wrapper para DELETE con manejo de errores y RETRY
inline httplib::Result hacerDELETE(const std::string& endpoint, int timeoutSegundos = 15) {
    auto [host, port] = obtenerPostgRESTConfig();
    std::cout << "[HTTP DEBUG] DELETE " << endpoint << " -> " << host << ":" << port << std::endl;
    
    auto headers = headersDELETE();
    
    // Retry logic: 3 intentos con delay incremental
    const int MAX_RETRIES = 3;
    httplib::Result res;
    
    for (int intento = 1; intento <= MAX_RETRIES; ++intento) {
        httplib::Client cli(host.c_str(), port);
        
        // CRITICO: keep_alive=false y write_timeout para Docker
        cli.set_keep_alive(false);
        cli.set_connection_timeout(timeoutSegundos, 0);
        cli.set_read_timeout(timeoutSegundos, 0);
        cli.set_write_timeout(timeoutSegundos, 0);
        
        res = cli.Delete(endpoint.c_str(), headers);
        
        if (res) {
            if (intento > 1) {
                std::cout << "[HTTP DEBUG] DELETE exitoso en intento " << intento << std::endl;
            }
            return res;  // Exito
        }
        
        // Fallo - logging y retry
        std::cerr << "[HTTP DEBUG] DELETE intento " << intento << "/" << MAX_RETRIES 
                  << " fallo - Error: " << httplib::to_string(res.error()) << std::endl;
        
        if (intento < MAX_RETRIES) {
            int delayMs = intento * 500;  // 500ms, 1000ms
            std::cout << "[HTTP DEBUG] Reintentando en " << delayMs << "ms..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
    }
    
    std::cerr << "[HTTP ERROR] DELETE fallo despues de " << MAX_RETRIES << " intentos" << std::endl;
    return res;  // Retornar ultimo resultado (nullptr)
}

// Validar respuesta y parsear JSON (para GET)
inline bool procesarResponseGET(const httplib::Result& res, nlohmann::json& output) {
    if (!res) {
        std::cerr << "[HTTP ERROR] Response nullptr - No conexion" << std::endl;
        return false;
    }
    
    if (res->status != 200) {
        std::cerr << "[HTTP ERROR] Status " << res->status << ": " << res->body << std::endl;
        return false;
    }
    
    try {
        output = nlohmann::json::parse(res->body);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HTTP ERROR] Error parsing JSON: " << e.what() << std::endl;
        return false;
    }
}

// Validar respuesta para POST (espera 201)
inline bool procesarResponsePOST(const httplib::Result& res, nlohmann::json& output) {
    if (!res) {
        std::cerr << "[HTTP ERROR] Response nullptr - No conexion" << std::endl;
        return false;
    }
    
    if (res->status != 201) {
        std::cerr << "[HTTP ERROR] Status " << res->status << ": " << res->body << std::endl;
        return false;
    }
    
    if (!res->body.empty()) {
        try {
            output = nlohmann::json::parse(res->body);
        } catch (...) {
            // Si falla el parse, al menos tenemos status 201
        }
    }
    
    return true;
}

// Validar respuesta para PATCH/DELETE (espera 204)
inline bool procesarResponseNoContent(const httplib::Result& res) {
    if (!res) {
        std::cerr << "[HTTP ERROR] Response nullptr - No conexion" << std::endl;
        return false;
    }
    
    if (res->status != 204) {
        std::cerr << "[HTTP ERROR] Status " << res->status << ": " << res->body << std::endl;
        return false;
    }
    
    return true;
}

} // namespace HttpHelpers

#endif // HTTP_HELPERS_H
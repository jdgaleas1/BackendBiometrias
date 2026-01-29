#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <chrono>
#include "controller/usuario_controller.h"
#include "../utils/http_helpers.h"
#include "../external/httplib.h"
#include "../external/json.hpp"
#include "controller/frases_controller.h"
#include "../utils/config.h"  

using json = nlohmann::json;
namespace fs = std::filesystem;

// FUNCIÓN AUXILIAR: Guardar archivo temporal
std::string guardarArchivoTemporal(const httplib::MultipartFormData& file, const std::string& prefix) {
    std::string extension = ".bin";
    size_t dotPos = file.filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        extension = file.filename.substr(dotPos);
    }
    
    // Usar ruta temporal desde config
    std::string tempPath = obtenerRutaTempAudio() + prefix + "_" +
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
        extension;

    std::ofstream ofs(tempPath, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "! ERROR: No se pudo crear archivo temporal: " << tempPath << std::endl;
        return "";
    }
    
    ofs.write(file.content.data(), file.content.size());
    
    if (!ofs.good()) {
        std::cerr << "! ERROR: Fallo al escribir archivo temporal: " << tempPath << std::endl;
        ofs.close();
        return "";
    }
    
    ofs.close();
    
    std::cout << "   @ Archivo temporal guardado: " << tempPath 
              << " (" << file.content.size() << " bytes)" << std::endl;

    return tempPath;
}

// MAIN - SERVIDOR HTTP
int main() {
    std::cout <<  std::string(70, '-') << std::endl;
    std::cout << "-> SERVIDOR - SISTEMA BIOMETRICO DE VOZ y ASR <-" << std::endl << std::endl;

    // Crear directorio temporal si no existe
    fs::create_directories(obtenerRutaTempAudio());

    // INICIALIZAR CONTROLADORES
    auto usuarioController = std::make_unique<UsuarioController>();
    auto frasesController = std::make_unique<FrasesController>();

    // CREAR SERVIDOR HTTP
    httplib::Server svr;

    // Configurar tamaño máximo de payload (50MB para audios)
    svr.set_payload_max_length(50 * 1024 * 1024);

    // ENDPOINTS - VOZ (Usuarios)
    // POST /voz/autenticar - Autenticar usuario por voz (REQUIERE: audio, identificador, id_frase)
    svr.Post("/voz/autenticar", [&usuarioController](const httplib::Request& req, httplib::Response& res) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "  ENDPOINT: /voz/autenticar" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        try {
            // Validaciones de campos requeridos
            if (!req.has_file("audio") || !req.has_file("identificador") || !req.has_file("id_frase")) {
                json error;
                error["success"] = false;
                error["error"] = "Faltan campos requeridos: audio, identificador, id_frase";
                res.status = 400;
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }
            
            auto audioFile = req.get_file_value("audio");
            std::string identificador = req.get_file_value("identificador").content;
            int idFrase = std::stoi(req.get_file_value("id_frase").content);
            
            std::cout << "-> Identificador: " << identificador << std::endl;
            std::cout << "-> ID Frase: " << idFrase << std::endl;

            // Guardar archivo temporal
            std::string tempPath = guardarArchivoTemporal(audioFile, "auth");
            if (tempPath.empty()) {
                json error;
                error["success"] = false;
                error["error"] = "Error al guardar archivo temporal";
                res.status = 500;
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }
            
            // Extraer IP del cliente
            std::string ipCliente = req.get_header_value("X-Forwarded-For");
            if (ipCliente.empty()) ipCliente = req.get_header_value("X-Real-IP");
            if (ipCliente.empty()) ipCliente = req.remote_addr;
            std::string userAgent = req.get_header_value("User-Agent");
            
            // Autenticar (el servicio valida si el identificador existe)
            json result = usuarioController->autenticar(tempPath, identificador, idFrase, ipCliente, userAgent);
            
            // Log resultado
            if (result["success"] == true && result["authenticated"] == true) {
                std::cout << "-> AUTENTICACION EXITOSA" << std::endl;
                std::cout << "   Usuario ID: " << result["user_id"] << std::endl;
                std::cout << "   Confianza: " << result["confidence"] << "%" << std::endl;
            } else {
                std::cout << "-> AUTENTICACION DENEGADA" << std::endl;
            }
            std::cout << std::string(60, '=') << "\n" << std::endl;
            
            // Limpiar temporal
            fs::remove(tempPath);

            res.set_content(result.dump(), "application/json; charset=utf-8");

        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // POST /voz/registrar_biometria - Registrar biometría de usuario YA EXISTENTE
    svr.Post("/voz/registrar_biometria", [&usuarioController](const httplib::Request& req, httplib::Response& res) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "  ENDPOINT: /voz/registrar_biometria" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        try {
            // Obtener identificador
            std::string identificador = req.get_file_value("identificador").content;
            std::cout << "-> Identificador: " << identificador << std::endl;

            if (identificador.empty()) {
                json error;
                error["success"] = false;
                error["error"] = "Identificador requerido";
                res.status = 400;
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }

            // Validar audios
            auto audioFiles = req.get_file_values("audios");
            std::cout << "-> Audios recibidos: " << audioFiles.size() << std::endl;
            
            if (audioFiles.size() < 6) {
                json error;
                error["success"] = false;
                error["error"] = "Se requieren al menos 6 audios";
                res.status = 400;
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }

            // Guardar audios temporalmente
            std::vector<std::string> audioPaths;
            for (size_t i = 0; i < audioFiles.size(); ++i) {
                std::string tempPath = guardarArchivoTemporal(audioFiles[i], "reg_bio_" + std::to_string(i));
                if (tempPath.empty()) {
                    // Limpiar archivos ya guardados
                    for (const auto& path : audioPaths) {
                        if (!path.empty()) fs::remove(path);
                    }
                    json error;
                    error["success"] = false;
                    error["error"] = "Error al guardar archivos temporales";
                    res.status = 500;
                    res.set_content(error.dump(), "application/json; charset=utf-8");
                    return;
                }
                audioPaths.push_back(tempPath);
            }

            // Registrar biometría (el servicio valida usuario y registra credencial)
            json result = usuarioController->registrarBiometria(identificador, audioPaths);

            // Limpiar archivos temporales
            for (const auto& path : audioPaths) {
                fs::remove(path);
            }

            if (result["success"] == true) {
                std::cout << "-> BIOMETRIA REGISTRADA EXITOSAMENTE" << std::endl;
                std::cout << "   Identificador: " << identificador << std::endl;
                if (result.contains("user_id")) {
                    std::cout << "   ID Usuario: " << result["user_id"] << std::endl;
                }
            } else {
                std::cout << "-> ERROR EN REGISTRO BIOMETRICO" << std::endl;
                if (result.contains("error")) {
                    std::cout << "   Error: " << result["error"] << std::endl;
                }
            }
            std::cout << std::string(60, '=') << "\n" << std::endl;

            res.status = result["success"] == true ? 200 : 500;
            res.set_content(result.dump(), "application/json; charset=utf-8");

        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.status = 500;
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });



    // GET /voz/usuarios - Listar usuarios
    svr.Get("/voz/usuarios", [&usuarioController](const httplib::Request&, httplib::Response& res) {
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "  ENDPOINT: GET /voz/usuarios" << std::endl;
        
        try {
            json result = usuarioController->listarUsuarios();
            
            if (result["success"] == true) {
                std::cout << "-> Total usuarios registrados: " << result["total"] << std::endl;
            }
            std::cout << std::string(60, '-') << "\n" << std::endl;
            
            res.set_content(result.dump(), "application/json; charset=utf-8");
        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // DELETE /voz/usuarios/:id - Eliminar usuario y su modelo biométrico
    svr.Delete("/voz/usuarios/:id", [&usuarioController](const httplib::Request& req, httplib::Response& res) {
        try {
            int userId = std::stoi(req.path_params.at("id"));
            
            std::cout << "\n" << std::string(60, '!') << std::endl;
            std::cout << "  ENDPOINT: DELETE /voz/usuarios/:id" << std::endl;
            std::cout << "  Usuario ID a eliminar: " << userId << std::endl;
            std::cout << std::string(60, '!') << std::endl;
            
            json result = usuarioController->eliminarUsuario(userId);
            
            if (result["success"] == true) {
                std::cout << "-> USUARIO ELIMINADO CORRECTAMENTE" << std::endl;
            } else {
                std::cout << "-> ERROR AL ELIMINAR USUARIO" << std::endl;
            }
            std::cout << std::string(60, '!') << "\n" << std::endl;
            
            res.set_content(result.dump(), "application/json; charset=utf-8");

        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // PATCH /voz/credenciales/:id/estado - Desactivar/Activar credencial biometrica
    svr.Patch("/voz/credenciales/:id/estado", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "\n" << std::string(60, '@') << std::endl;
        std::cout << "  ENDPOINT: PATCH /voz/credenciales/:id/estado" << std::endl;
        std::cout << std::string(60, '@') << std::endl;
        
        try {
            int id_credencial = std::stoi(req.path_params.at("id"));
            auto body = json::parse(req.body);
            std::string nuevo_estado = body["estado"];
            
            std::cout << "-> Actualizando credencial ID " << id_credencial 
                      << " a estado: " << nuevo_estado << std::endl;
            
            if (nuevo_estado != "activo" && nuevo_estado != "inactivo") {
                json error;
                error["success"] = false;
                error["error"] = "Estado invalido (debe ser 'activo' o 'inactivo')";
                res.status = 400;
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }
            
            // Actualizar en PostgreSQL via PostgREST
            json update_data = {
                {"estado", nuevo_estado}
            };
            
            std::string url = "/credenciales_biometricas?id_credencial=eq." + std::to_string(id_credencial);
            auto res_update = HttpHelpers::hacerPATCH(url, update_data, 15);
            
            if (!res_update || (res_update->status != 200 && res_update->status != 204)) {
                std::cerr << "! ERROR: No se pudo actualizar credencial en BD" << std::endl;
                if (res_update) {
                    std::cerr << "   Status: " << res_update->status << std::endl;
                    std::cerr << "   Body: " << res_update->body << std::endl;
                }
                json error;
                error["success"] = false;
                error["error"] = "No se pudo actualizar credencial en BD";
                res.status = 500;
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }
            
            std::cout << "-> CREDENCIAL ACTUALIZADA CORRECTAMENTE" << std::endl;
            std::cout << std::string(60, '@') << "\n" << std::endl;
            
            json response;
            response["success"] = true;
            response["message"] = "Credencial actualizada correctamente";
            response["id_credencial"] = id_credencial;
            response["nuevo_estado"] = nuevo_estado;
            
            res.set_content(response.dump(), "application/json; charset=utf-8");
            
        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.status = 500;
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // ENDPOINTS - FRASES (Textos dinámicos)

    // POST /agregar/frases - Agregar nueva frase al sistema
    svr.Post("/agregar/frases", [&frasesController](const httplib::Request& req, httplib::Response& res) {
        std::cout << "\n" << std::string(60, '+') << std::endl;
        std::cout << "  ENDPOINT: POST /agregar/frases" << std::endl;
        std::cout << std::string(60, '+') << std::endl;
        
        try {
            auto body = json::parse(req.body);
            std::string frase = body["frase"];
            
            std::cout << "-> Nueva frase a agregar: \"" << frase << "\"" << std::endl;

            if (frase.empty()) {
                json error;
                error["success"] = false;
                error["error"] = "Frase vacia";
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }

            json result = frasesController->agregarFrase(frase);
            
            if (result["success"] == true) {
                std::cout << "-> FRASE AGREGADA EXITOSAMENTE" << std::endl;
                if (result.contains("id_texto")) {
                    std::cout << "   ID Frase: " << result["id_texto"] << std::endl;
                }
            }
            std::cout << std::string(60, '+') << "\n" << std::endl;
            
            res.set_content(result.dump(), "application/json; charset=utf-8");

        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // GET /frases/aleatoria - Obtener frase aleatoria activa
    svr.Get("/frases/aleatoria", [&frasesController](const httplib::Request&, httplib::Response& res) {
        std::cout << "-> Solicitando frase aleatoria..." << std::endl;
        
        try {
            json result = frasesController->obtenerFraseAleatoria();
            
            if (result["success"] == true && result.contains("frase")) {
                std::cout << "   Frase: \"" << result["frase"] << "\"" << std::endl;
            }
            
            res.set_content(result.dump(), "application/json; charset=utf-8");
        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // GET /listar/frases - Listar todas las frases (o específica por ID)
    svr.Get("/listar/frases", [&frasesController](const httplib::Request& req, httplib::Response& res) {
        try {
            if (req.has_param("id")) {
                int id = std::stoi(req.get_param_value("id"));
                std::cout << "-> Consultando frase ID: " << id << std::endl;
                json result = frasesController->obtenerFrasePorId(id);
                res.set_content(result.dump(), "application/json; charset=utf-8");
            } else {
                std::cout << "-> Listando todas las frases..." << std::endl;
                json result = frasesController->listarFrases();
                if (result["success"] == true && result.contains("total")) {
                    std::cout << "   Total frases: " << result["total"] << std::endl;
                }
                res.set_content(result.dump(), "application/json; charset=utf-8");
            }
        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // PATCH /frases/:id/estado - Activar/Desactivar frase
    svr.Patch("/frases/:id/estado", [&frasesController](const httplib::Request& req, httplib::Response& res) {
        try {
            int id_texto = std::stoi(req.path_params.at("id"));
            auto body = json::parse(req.body);

            int activo = body["activo"];
            
            std::cout << "-> Actualizando estado de frase ID " << id_texto 
                     << " a: " << (activo ? "ACTIVA" : "INACTIVA") << std::endl;
            
            if (activo != 0 && activo != 1) {
                json error;
                error["success"] = false;
                error["error"] = "Valor invalido para 'activo' (debe ser 0 o 1)";
                res.set_content(error.dump(), "application/json; charset=utf-8");
                return;
            }

            json result = frasesController->actualizarEstadoFrase(id_texto, activo);
            res.set_content(result.dump(), "application/json; charset=utf-8");

        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // DELETE /frases/:id - Eliminar frase
    svr.Delete("/frases/:id", [&frasesController](const httplib::Request& req, httplib::Response& res) {
        try {
            int id_texto = std::stoi(req.path_params.at("id"));
            
            std::cout << "-> Eliminando frase ID: " << id_texto << std::endl;
            
            json result = frasesController->eliminarFrase(id_texto);
            res.set_content(result.dump(), "application/json; charset=utf-8");

        } catch (const std::exception& e) {
            json error;
            error["success"] = false;
            error["error"] = std::string("Error: ") + e.what();
            res.set_content(error.dump(), "application/json; charset=utf-8");
        }
    });

    // CONFIGURACIÓN CORS Y OPTIONS
    // Configurar CORS para permitir peticiones desde cualquier origen
    svr.set_pre_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // Manejar peticiones OPTIONS (preflight CORS)
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
    });

    // INICIAR SERVIDOR
    std::cout << "\n-> Servidor biometrico de la voz activo en http://0.0.0.0:8081 \n"  << std::endl;
    svr.listen("0.0.0.0", 8081);
    return 0;
}

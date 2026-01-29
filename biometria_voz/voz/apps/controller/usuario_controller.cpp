#include "usuario_controller.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <regex>
#include "../../utils/config.h"
#include "../../external/httplib.h"

namespace fs = std::filesystem;

UsuarioController::UsuarioController() {

    // ✅ USAR RUTAS DESDE CONFIG.H
    modelPath = obtenerRutaModelo(); 
    mappingPath = "";  // Ya no se usa, metadata.json es la fuente de verdad
    trainDataPath = obtenerRutaDatasetTrain();
    tempDir = obtenerRutaTempAudio();

    // Crear directorio temporal
    fs::create_directories(tempDir);

    // Inicializar servicios (mappingPath se ignora ahora)
    authService = std::make_unique<AutenticacionService>(modelPath, mappingPath);
    registerService = std::make_unique<RegistrarService>(mappingPath, trainDataPath);
    listService = std::make_unique<ListarService>(mappingPath);

    // MOSTRAR RUTAS CONFIGURADAS
    std::cout << "-> Rutas configuradas:" << std::endl;
    std::cout << "      Modelo:   " << modelPath << std::endl;
    std::cout << "      Mapeo:  " << trainDataPath << std::endl;
    std::cout << "      Temp:     " << tempDir << std::endl;
    std::cout << "   Nota: Mapeo de usuarios gestionado por metadata.json" << std::endl;
}

UsuarioController::~UsuarioController() {
    // Limpiar directorio temporal
    try {
        fs::remove_all(tempDir);
    }
    catch (...) {}
}

json UsuarioController::autenticar(const std::string& audioPath, const std::string& identificador, int idFrase,
                                        const std::string& ipCliente, const std::string& userAgent) {
    json response;

    try {
        auto resultado = authService->autenticar(audioPath, identificador, idFrase);

        response["success"] = resultado.exito;

        if (resultado.exito) {
            // VALIDACION CRITICA: Verificar que el identificador coincida con el resultado del SVM
            std::string idDetectado = std::to_string(resultado.userId);
            bool identificadorCoincide = (identificador == idDetectado);
            
            // Solo AUTORIZAR si ambas condiciones se cumplen:
            // 1. SVM autentico correctamente
            // 2. El identificador proporcionado coincide con el detectado
            bool autenticadoFinal = resultado.autenticado && identificadorCoincide;
            
            response["authenticated"] = autenticadoFinal;
            response["access"] = autenticadoFinal;
            response["user_id"] = resultado.userId;
            response["user_name"] = resultado.userName;
            response["confidence"] = resultado.confianza;
            response["processing_time_ms"] = resultado.tiempoProcesamiento;
            response["identificador_esperado"] = identificador;
            response["identificador_detectado"] = idDetectado;
            response["identificador_coincide"] = identificadorCoincide;
            
            // Agregar info de texto dinámico
            response["frase_esperada"] = resultado.fraseEsperada;
            response["transcripcion"] = resultado.transcripcionDetectada;
            response["similitud_texto"] = resultado.similitudTexto;
            response["texto_coincide"] = resultado.textoCoincide;
            
            // Agregar scores detallados
            json scoresJson;
            for (const auto& [id, score] : resultado.scores) {
                scoresJson[std::to_string(id)] = score;
            }
            response["all_scores"] = scoresJson;

            if (!identificadorCoincide) {
                std::cout << "-> Auth: DENEGADO - Identificador no coincide" << std::endl;
                std::cout << "   Esperado: " << identificador << " | Detectado: " << idDetectado << std::endl;
            } else {
                std::cout << "-> Auth: ID " << resultado.userId << " - "
                    << (autenticadoFinal ? "AUTORIZADO" : "DENEGADO")
                    << " (conf: " << resultado.confianza << ")" << std::endl;
            }
        }
        else {
            response["error"] = resultado.error;
        }

    }
    catch (const std::exception& e) {
        response["success"] = false;
        response["error"] = std::string("Excepcion en controlador: ") + e.what();
    }

    return response;
}

json UsuarioController::registrarUsuario(const std::string& nombre,
    const std::vector<std::string>& audiosPaths) {
    json response;

    try {
        auto resultado = registerService->registrarUsuario(nombre, audiosPaths);

        response["success"] = resultado.exito;

        if (resultado.exito) {
            response["user_id"] = resultado.userId;
            response["user_name"] = resultado.userName;
            response["total_audios"] = resultado.totalAudios;
            response["audios_exitosos"] = resultado.audiosExitosos;
            response["audios_fallidos"] = resultado.audiosFallidos;
            response["message"] = "Usuario registrado exitosamente.";

            std::cout << "-> Usuario registrado: " << nombre
                << " (ID: " << resultado.userId << ")" << std::endl;

            // ENTRENAR MODELO UNA SOLA VEZ AQUÍ
            std::cout << "&  Reentrenando modelo de voz..." << std::endl;
            auto resultadoEntrenamiento = registerService->entrenarModelo();

            if (!resultadoEntrenamiento.exito) {
                response["warning"] = "Registro completado, pero fallo el entrenamiento";
                response["train_error"] = resultadoEntrenamiento.error;
            }
            else {
                response["training_message"] = resultadoEntrenamiento.mensaje;
                response["num_classes"] = resultadoEntrenamiento.numClases;

                // RECARGAR CONFIGURACIÓN DESPUeS DEL ENTRENAMIENTO
                recargarConfiguracion();
            }
        }
        else {
            response["error"] = resultado.error;
        }

    }
    catch (const std::exception& e) {
        response["success"] = false;
        response["error"] = std::string("Excepcion en controlador: ") + e.what();
    }

    return response;
}

json UsuarioController::entrenarModelo() {
    json response;

    try {
        std::cout << "-> Iniciando entrenamiento del modelo..." << std::endl;

        auto resultado = registerService->entrenarModelo();

        response["success"] = resultado.exito;

        if (resultado.exito) {
            response["message"] = resultado.mensaje;
            response["num_classes"] = resultado.numClases;

            // Recargar configuración
            recargarConfiguracion();

            std::cout << "-> Modelo entrenado con " << resultado.numClases
                << " clases" << std::endl;
        }
        else {
            response["error"] = resultado.error;
        }

    }
    catch (const std::exception& e) {
        response["success"] = false;
        response["error"] = std::string("Excepcion en controlador: ") + e.what();
    }

    return response;
}

json UsuarioController::listarUsuarios() {
    json response;

    try {
        auto resultado = listService->listarUsuarios();

        response["success"] = resultado.exito;

        if (resultado.exito) {
            json usuarios = json::array();

            for (const auto& usuario : resultado.usuarios) {
                json u;
                u["id"] = usuario.id;
                u["nombre"] = usuario.nombre;
                usuarios.push_back(u);
            }

            response["total"] = resultado.total;
            response["usuarios"] = usuarios;
        }
        else {
            response["error"] = resultado.error;
        }

    }
    catch (const std::exception& e) {
        response["success"] = false;
        response["error"] = std::string("Excepcion en controlador: ") + e.what();
    }

    return response;
}

json UsuarioController::eliminarUsuario(int userId) {
    json response;

    try {
        bool exito = listService->eliminarUsuario(userId);

        if (exito) {
            response["success"] = true;
            response["message"] = "Usuario y su modelo biometrico eliminados correctamente";
            response["user_id"] = userId;

            // Recargar servicios para actualizar modelos en memoria
            authService->recargarModelo(modelPath);
            authService->recargarMapeos(mappingPath);
        } else {
            response["success"] = false;
            response["error"] = "No se pudo eliminar el usuario";
        }
    }
    catch (const std::exception& e) {
        response["success"] = false;
        response["error"] = std::string("Error: ") + e.what();
    }

    return response;
}

json UsuarioController::registrarBiometria(const std::string& cedula, const std::vector<std::string>& audioPaths) {
    std::cout << "\n-> Controller: Registrando biometria para cedula: " << cedula << std::endl;
    
    try {
        // Delegar al servicio de registro usando registerService directamente
        json resultado = registerService->registrarBiometriaPorCedula(cedula, audioPaths);
        
        // Si el registro fue exitoso, entrenar modelo
        if (resultado["success"] == true) {
            std::cout << std::endl << std::string(70, '-') << std::endl;
            std::cout << "-> COMIENZA ETAPA DE ENTRENAMIENTO DEL MODELO SVM..." << std::endl;
            auto resultadoEntrenamiento = registerService->entrenarModelo();
            
            if (!resultadoEntrenamiento.exito) {
                resultado["warning"] = "Biometria registrada pero fallo el entrenamiento";
                resultado["train_error"] = resultadoEntrenamiento.error;
            } else {
                resultado["training_message"] = resultadoEntrenamiento.mensaje;
                resultado["num_classes"] = resultadoEntrenamiento.numClases;
                
                // Recargar configuración
                recargarConfiguracion();
            }
        }
        
        return resultado;
        
    } catch (const std::exception& e) {
        std::cerr << "# ERROR en registrarBiometria: " << e.what() << std::endl;
        json error;
        error["success"] = false;
        error["error"] = std::string("Error al registrar biometria: ") + e.what();
        return error;
    }
}

void UsuarioController::recargarConfiguracion() {
    std::cout << "-> Recargando configuración en todos los servicios..." << std::endl;

    authService->recargarModelo(modelPath);
    authService->recargarMapeos("");  // Ahora lee de metadata.json
    listService->recargarDatos("");   // Ahora lee de metadata.json

    std::cout << "-> Configuración recargada exitosamente" << std::endl;
}

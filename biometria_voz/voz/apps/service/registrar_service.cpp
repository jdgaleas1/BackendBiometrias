#include <utility>
#include "../../utils/http_helpers.h"
#include "registrar_service.h"
#include "../../external/httplib.h"
#include "../../external/json.hpp"
#include "../../core/pipeline/audio_pipeline.h"
#include "../../core/classification/svm.h"
#include "../../core/classification/training/svm_training.h" 
#include "../../core/process_dataset/dataset.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <set>
#include <random>

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace HttpHelpers;  // Para usar los helpers HTTP

// Estructura para resumen del registro
struct RegistroResumen {
    int idUsuario;
    std::string nombreUsuario;
    int totalAudios;
    int audiosExitosos;
    int audiosFallidos;
    bool exito;
    std::string error;
};

// Funciones de normalización y augmentation ahora están en core/pipeline/

void generarReporteRegistro(const RegistroResumen& resumen, const std::string& datasetPath) {
    std::filesystem::path datasetFile(datasetPath);
    std::filesystem::path outputDir = datasetFile.parent_path();
    std::string reportePath = (outputDir / "registro_usuario_report.txt").string();

    std::ofstream reporte(reportePath, std::ios::app);
    if (!reporte.is_open()) {
        std::cerr << "% Error: No se pudo crear el reporte de registro" << std::endl;
        return;
    }

    auto now = std::time(nullptr);
    std::tm tm_buf;
    std::tm* tm;

#ifdef _WIN32
    if (localtime_s(&tm_buf, &now) == 0) tm = &tm_buf;
    else tm = nullptr;
#else
    tm = localtime_r(&now, &tm_buf);
#endif

    reporte << std::string(60, '=') << "\n";

    if (tm != nullptr)
        reporte << "Fecha: " << std::put_time(tm, "%Y-%m-%d %H:%M:%S") << "\n";
    else
        reporte << "Fecha: Desconocida\n";

    reporte << "ID Usuario:      " << resumen.idUsuario << "\n";
    reporte << "Nombre Usuario:  " << resumen.nombreUsuario << "\n";
    reporte << "Audios recibidos: " << resumen.totalAudios << "\n";
    reporte << "Audios exitosos:  " << resumen.audiosExitosos << "\n";
    reporte << "Audios fallidos:  " << resumen.audiosFallidos << "\n";

    if (resumen.exito)
        reporte << "\nEstado: REGISTRO EXITOSO\n";
    else
        reporte << "\nEstado: REGISTRO FALLIDO\nError: " << resumen.error << "\n";

    reporte << std::string(60, '=') << "\n";
    reporte.close();

    std::cout << "&  Reporte de registro generado en: " << reportePath << std::endl;
}

RegistrarService::RegistrarService(const std::string& mappingPath,
    const std::string& trainDataPath)
    : mappingPath(mappingPath), trainDataPath(trainDataPath) {
    cargarMapeos();
}

void RegistrarService::cargarMapeos() {
    mapeoUsuarios.clear();

    std::string metadataPath = obtenerRutaModelo() + "metadata.json";
    std::ifstream metaIn(metadataPath);
    
    if (metaIn.is_open()) {
        try {
            json metadata;
            metaIn >> metadata;
            metaIn.close();

            auto clases = metadata["classes"];
            
            for (int userId : clases) {
                mapeoUsuarios[userId] = "Usuario_" + std::to_string(userId);
            }

        } catch (const std::exception& e) {
            std::cerr << "! Error parseando metadata.json: " << e.what() << std::endl;
        }
    }
}

bool RegistrarService::procesarAudio(const std::string& audioPath, std::vector<AudioSample>& features) {
    features.clear();

    std::vector<std::vector<AudioSample>> todasFeatures;
    if (!procesarAudioCompleto(audioPath, todasFeatures) || todasFeatures.empty()) {
        std::cerr << "   ! Pipeline fallo: no se generaron features" << std::endl;
        return false;
    }

    features = todasFeatures[0];
    
    // Validar dimension esperada segun configuracion
    size_t expectedFeatures = CONFIG_MFCC.totalFeatures;
    if (CONFIG_SVM.usarExpansionPolinomial) {
        expectedFeatures *= 2;
    }
    
    if (features.size() != expectedFeatures) {
        std::cerr << "! Error: Dimension incorrecta - esperado: " << expectedFeatures 
                  << ", obtenido: " << features.size() << std::endl;
        return false;
    }
    
    std::cout << "   @ Pipeline exitoso: " << features.size() << " features extraidos" << std::endl;
    return true;
}

int RegistrarService::obtenerSiguienteId() {
    if (mapeoUsuarios.empty()) {
        return 1;
    }
    return mapeoUsuarios.rbegin()->first + 1;
}

ResultadoRegistro RegistrarService::registrarUsuario(const std::string& nombre, const std::vector<std::string>& audiosPaths) {
    ResultadoRegistro resultado;
    resultado.userName = nombre;
    resultado.totalAudios = audiosPaths.size();

    std::vector<std::vector<AudioSample>> featuresList;
    
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "PROCESAMIENTO DE AUDIOS" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    // USAR PIPELINE que ya maneja augmentation automaticamente
    for (size_t i = 0; i < audiosPaths.size(); ++i) {
        const auto& path = audiosPaths[i];
        
        std::cout << "\n[" << (i+1) << "/" << audiosPaths.size() << "] Procesando audio..." << std::endl;
        
        std::vector<std::vector<AudioSample>> audioFeatures;
        if (!procesarAudioCompleto(path, audioFeatures) || audioFeatures.empty()) {
            std::cerr << "   ! Audio rechazado\n" << std::endl;
            resultado.audiosFallidos++;
            continue;
        }
        
        // El pipeline ya retorna augmentation si esta configurado
        // audioFeatures contiene [original + variaciones]
        for (const auto& features : audioFeatures) {
            featuresList.push_back(features);
        }
        
        std::cout << "   @ Audio [" << (i+1) << "/" << audiosPaths.size() << "] procesado correctamente\n" << std::endl;
        resultado.audiosExitosos++;
    }
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "RESUMEN PROCESAMIENTO" << std::endl;
    std::cout << "   Audios exitosos: " << resultado.audiosExitosos << "/" << resultado.totalAudios << std::endl;
    std::cout << "   Grabaciones procesadas: " << featuresList.size() << " -> " << featuresList.size() << " ejemplos de entrenamiento" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    if (featuresList.size() < 6) {
        std::cerr << "\n! Error: Se requieren minimo 6 grabaciones (" << featuresList.size() << " obtenidas)" << std::endl;
        resultado.exito = false;
        resultado.error = "Menos de 6 grabaciones validas procesadas.";
        return resultado;
    }

    // Obtener nuevo ID
    int nuevoId = obtenerSiguienteId();
    resultado.userId = nuevoId;

    // ✅ AGREGAR EJEMPLOS INCREMENTALMENTE (sin recargar todo el dataset)
    std::vector<int> labels(featuresList.size(), nuevoId);
    
    if (!agregarMuestrasDataset(trainDataPath, featuresList, labels)) {
        resultado.exito = false;
        resultado.error = "No se pudo agregar ejemplos al dataset";
        return resultado;
    }

    // Ya no actualizamos speaker_mapping.txt (deprecated)
    // El mapeo se gestiona automáticamente en metadata.json durante el entrenamiento

    resultado.exito = true;

    std::cout << "-> Usuario " << nombre << " registrado con ID " << nuevoId
        << " (" << featuresList.size() << " grabaciones -> " << featuresList.size() << " ejemplos)" << std::endl;

    //  NO ENTRENAR AQUÍ - Se hace en el controlador
    return resultado;
}

ResultadoEntrenamientoModelo RegistrarService::entrenarModelo() {
    ResultadoEntrenamientoModelo resultado;

    try {
        std::vector<std::vector<AudioSample>> X_train;
        std::vector<int> y_train;

        if (!cargarDatasetBinario(trainDataPath, X_train, y_train)) {
            resultado.exito = false;
            resultado.error = "No se pudo cargar las caracteristicas de las clases";
            return resultado;
        }

        // ====================================================================
        // ENTRENAMIENTO INTELIGENTE
        // ====================================================================
        // - Primera vez: Entrenamiento completo (todos los clasificadores)
        // - Agregar usuario: Entrenamiento incremental BALANCEADO
        //   * Submuestreo de negativas para igualar ratio de modelos existentes
        //   * Ajuste de bias basado en estadisticas de clasificadores previos
        //   * Rapido (~10 seg) vs completo (~2 min)
        // ====================================================================
        std::string modeloBase = obtenerRutaModelo();
        std::string metadataPath = modeloBase + "metadata.json";
        bool modelo_existe = fs::exists(metadataPath);

        // Contar clases en el dataset
        std::map<int, int> ejemplos_por_clase;
        for (int label : y_train) {
            ejemplos_por_clase[label]++;
        }
        int num_clases_dataset = static_cast<int>(ejemplos_por_clase.size());

        if (!modelo_existe) {
            // ENTRENAMIENTO INICIAL 
            std::cout << "\n========================================" << std::endl;
            std::cout << "   ENTRENAMIENTO INICIAL" << std::endl;
            std::cout << "   Creando modelo desde cero" << std::endl;
            std::cout << "========================================" << std::endl;
            
            std::cout << "\n-> Caracteristicas cargadas:" << std::endl;
            std::cout << "   Total ejemplos: " << X_train.size() << std::endl;
            std::cout << "   Clases detectadas: " << num_clases_dataset << std::endl;
            std::cout << "   Dimension: " << X_train[0].size() << " features" << std::endl;
            
            // Entrenar modelo completo
            ModeloSVM modelo = entrenarSVMOVA(X_train, y_train);

            // Guardar en formato modular
            if (!guardarModeloModular(modeloBase, modelo)) {
                resultado.exito = false;
                resultado.error = "No se pudo guardar el modelo modular";
                return resultado;
            }

            resultado.exito = true;
            resultado.mensaje = "Modelo inicial entrenado";
            resultado.numClases = modelo.clases.size();

            std::cout << "\n-> Modelo guardado exitosamente:" << std::endl;
            std::cout << "   Clases totales: " << resultado.numClases << std::endl;
            std::cout << "   Ruta: " << modeloBase << std::endl;
            
        } else {
            // ENTRENAMIENTO INCREMENTAL BALANCEADO (agregar usuario)
            std::cout << "\n========================================" << std::endl;
            std::cout << "   ENTRENAMIENTO INCREMENTAL BALANCEADO" << std::endl;

            // Detectar cual es la clase NUEVA (NO existe en modelo actual)
            ModeloSVM modelo_actual = cargarModeloModular(modeloBase);
            
            if (modelo_actual.clases.empty()) {
                resultado.exito = false;
                resultado.error = "No se pudo cargar modelo existente para deteccion incremental";
                return resultado;
            }

            // Crear set de clases existentes en el modelo
            std::set<int> clases_existentes(modelo_actual.clases.begin(), modelo_actual.clases.end());
            
            // Detectar clases nuevas (presentes en dataset pero NO en modelo)
            int clase_nueva = -1;
            for (const auto& [clase, count] : ejemplos_por_clase) {
                if (clases_existentes.find(clase) == clases_existentes.end()) {
                    clase_nueva = clase;
                    break;
                }
            }

            if (clase_nueva == -1) {
                resultado.exito = false;
                resultado.error = "No se encontro una clase nueva para entrenar incrementalmente";
                return resultado;
            }

            std::cout << "\n-> Caracteristicas cargadas:" << std::endl;
            std::cout << "   Total ejemplos: " << X_train.size() << std::endl;
            std::cout << "   Clases detectadas: " << num_clases_dataset << std::endl;
            std::cout << "   Clases en modelo existente: " << modelo_actual.clases.size() << std::endl;
            std::cout << "   Nueva clase detectada: " << clase_nueva << std::endl;
            std::cout << "   Dimension: " << X_train[0].size() << " features" << std::endl;
            
            // Mostrar distribucion por clase
            std::cout << "\n-> Distribucion por clase:" << std::endl;
            for (const auto& [clase, count] : ejemplos_por_clase) {
                std::cout << "   Clase " << clase << ": " << count << " ejemplos" 
                          << (clase == clase_nueva ? " <-- NUEVA" : "") << std::endl;
            }

            // Entrenar SOLO esa clase con balance inteligente
            if (!entrenarClaseIncremental(modeloBase, X_train, y_train, clase_nueva)) {
                resultado.exito = false;
                resultado.error = "No se pudo entrenar la clase incremental";
                return resultado;
            }

            resultado.exito = true;
            resultado.mensaje = "Usuario agregado incrementalmente (balanceado)";
            
            // Contar clases totales desde metadata
            int num_clases_total, dimension;
            std::vector<int> clases;
            if (cargarMetadata(modeloBase, num_clases_total, dimension, clases)) {
                resultado.numClases = num_clases_total;
            } else {
                resultado.numClases = 0; // Error al leer metadata
            }

            std::cout << "\n-> Clase " << clase_nueva << " agregada. Total: "
                      << resultado.numClases << " clases" << std::endl;
        }

    } catch (const std::exception& e) {
        resultado.exito = false;
        resultado.error = std::string("Excepción: ") + e.what();
    }

    return resultado;
}

json RegistrarService::registrarUsuarioCompleto(const std::string& nombre, const std::vector<std::string>& audioPaths) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  REGISTRO COMPLETO DE USUARIO" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "-> Nombre: " << nombre << std::endl;
    std::cout << "-> Audios proporcionados: " << audioPaths.size() << std::endl;

    json response;

    try {
        std::vector<std::vector<AudioSample>> featuresList;
        int audiosExitosos = 0;
        int audiosFallidos = 0;

        // Procesar cada audio
        for (size_t i = 0; i < audioPaths.size(); ++i) {
            std::vector<std::vector<AudioSample>> audioFeatures;
            if (!procesarAudioCompleto(audioPaths[i], audioFeatures) || audioFeatures.empty()) {
                audiosFallidos++;
                std::cerr << "   # Error procesando audio " << (i+1) << std::endl;
                continue;
            }
            
            for (const auto& features : audioFeatures) {
                featuresList.push_back(features);
            }
            audiosExitosos++;
            std::cout << "   * Audio " << (i+1) << "/" << audioPaths.size() 
                     << " procesado correctamente" << std::endl;
        }

        if (featuresList.size() < 6) {
            response["success"] = false;
            response["error"] = "Se requieren minimo 6 audios validos procesados";
            response["audios_exitosos"] = audiosExitosos;
            response["audios_fallidos"] = audiosFallidos;
            return response;
        }

        // Obtener nuevo ID
        int nuevoId = obtenerSiguienteId();
        
        // Agregar muestras al dataset
        std::vector<int> labels(featuresList.size(), nuevoId);
        
        if (!agregarMuestrasDataset(trainDataPath, featuresList, labels)) {
            response["success"] = false;
            response["error"] = "No se pudo agregar muestras al dataset";
            return response;
        }

        // Actualizar mapeo en memoria
        mapeoUsuarios[nuevoId] = nombre;

        response["success"] = true;
        response["message"] = "Usuario registrado exitosamente";
        response["user_id"] = nuevoId;
        response["nombre"] = nombre;
        response["muestras_procesadas"] = featuresList.size();
        response["audios_exitosos"] = audiosExitosos;
        response["audios_fallidos"] = audiosFallidos;

        std::cout << "\n-> Usuario registrado con ID " << nuevoId 
                 << " (" << featuresList.size() << " muestras)" << std::endl;
        std::cout << "========================================\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n# ERROR CRITICO: " << e.what() << std::endl;
        response["success"] = false;
        response["error"] = std::string("Error en registro: ") + e.what();
    }

    return response;
}

json RegistrarService::registrarBiometriaPorCedula(const std::string& cedula, const std::vector<std::string>& audioPaths) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  REGISTRO DE BIOMETRIA POR CEDULA" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "-> Cedula: " << cedula << std::endl;
    std::cout << "-> Audios proporcionados: " << audioPaths.size() << std::endl;

    json response;

    try {
        // 1. Verificar que usuario existe en BD y obtener id_usuario
        std::cout << "\n[DEBUG] === VALIDACION DE USUARIO EN BD ===" << std::endl;
        std::cout << "[DEBUG] Cedula a buscar: " << cedula << std::endl;
        
        std::string endpoint = "/usuarios?identificador_unico=eq." + cedula;
        std::cout << "[DEBUG] Endpoint GET: " << endpoint << std::endl;
        std::cout << "[DEBUG] Ejecutando GET..." << std::endl;
        
        auto res = HttpHelpers::hacerGET(endpoint, 15);
        
        if (!res) {
            std::cerr << "[ERROR] Response es nullptr - No se pudo conectar a PostgREST" << std::endl;
            response["success"] = false;
            response["error"] = "No se pudo conectar a la base de datos";
            return response;
        }
        
        std::cout << "[DEBUG] Response recibido - Status: " << res->status << std::endl;
        
        if (res->status != 200) {
            std::cerr << "[ERROR] Status HTTP no es 200: " << res->status << std::endl;
            response["success"] = false;
            response["error"] = "Usuario no registrado en el sistema";
            return response;
        }
        
        json usuarios;
        try {
            std::cout << "[DEBUG] Parseando JSON..." << std::endl;
            usuarios = json::parse(res->body);
            std::cout << "[DEBUG] JSON parseado - Array size: " << usuarios.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Error parsing JSON: " << e.what() << std::endl;
            response["success"] = false;
            response["error"] = "Error parsing respuesta de BD";
            return response;
        }
        
        if (usuarios.empty()) {
            std::cerr << "[ERROR] Array de usuarios vacio - Usuario no existe" << std::endl;
            response["success"] = false;
            response["error"] = "Usuario no registrado en el sistema";
            return response;
        }
        
        int idUsuario = usuarios[0]["id_usuario"];
        std::cout << "[DEBUG] Usuario encontrado - ID: " << idUsuario << std::endl;
        std::cout << "[DEBUG] === FIN VALIDACION USUARIO ===" << std::endl;
        
        // 1.5. Verificar si YA tiene credencial de voz registrada (evitar duplicados)
        std::cout << "\n[DEBUG] === VERIFICACION CREDENCIAL EXISTENTE ===" << std::endl;
        std::string endpointCred = "/credenciales_biometricas?id_usuario=eq." + std::to_string(idUsuario) + 
                                    "&tipo_biometria=eq.voz";
        std::cout << "[DEBUG] Buscando credenciales: " << endpointCred << std::endl;
        
        auto resCred = HttpHelpers::hacerGET(endpointCred, 15);
        if (resCred && resCred->status == 200) {
            json credencialesExistentes = json::parse(resCred->body);
            if (!credencialesExistentes.empty()) {
                std::cerr << "[ERROR] Usuario ya tiene credencial de voz registrada" << std::endl;
                response["success"] = false;
                response["error"] = "El usuario ya tiene biometria de voz registrada";
                response["credencial_existente"] = credencialesExistentes[0];
                return response;
            }
        }
        std::cout << "[DEBUG] No hay credencial previa - OK para continuar" << std::endl;
        std::cout << "[DEBUG] === FIN VERIFICACION CREDENCIAL ===" << std::endl;
        
        // 2. Procesar audios
        std::vector<std::vector<AudioSample>> featuresList;
        int audiosExitosos = 0;
        int audiosFallidos = 0;

        // Procesar cada audio
        for (size_t i = 0; i < audioPaths.size(); ++i) {
            std::vector<std::vector<AudioSample>> audioFeatures;
            if (!procesarAudioCompleto(audioPaths[i], audioFeatures) || audioFeatures.empty()) {
                audiosFallidos++;
                std::cerr << "   # Error procesando audio " << (i+1) << std::endl;
                continue;
            }
            
            featuresList.push_back(audioFeatures[0]);
            
            audiosExitosos++;
            std::cout << "   * Audio " << (i+1) << "/" << audioPaths.size() 
                     << " procesado correctamente" << std::endl;
        }

        if (featuresList.size() < 6) {
            response["success"] = false;
            response["error"] = "Se requieren minimo 6 audios validos procesados";
            response["audios_exitosos"] = audiosExitosos;
            response["audios_fallidos"] = audiosFallidos;
            return response;
        }

        // 4. Usar identificador del usuario como ID (en vez de generar ID aleatorio)
        int nuevoId = std::stoi(cedula);
        
        // Agregar muestras al dataset
        std::vector<int> labels(featuresList.size(), nuevoId);
        
        if (!agregarMuestrasDataset(trainDataPath, featuresList, labels)) {
            response["success"] = false;
            response["error"] = "No se pudo agregar muestras al dataset";
            return response;
        }

        // Actualizar mapeo usando cedula como nombre
        mapeoUsuarios[nuevoId] = cedula;

        // 4. Registrar credencial en la base de datos (ANTES de entrenar)
        std::cout << "\n[DEBUG] === REGISTRO DE CREDENCIAL EN BD ===" << std::endl;
        
        json credencial = {
            {"id_usuario", idUsuario},
            {"tipo_biometria", "voz"},
            {"estado", "activo"}
        };
        
        std::cout << "[DEBUG] Credencial a insertar: " << credencial.dump(2) << std::endl;
        std::cout << "[DEBUG] Endpoint POST: /credenciales_biometricas" << std::endl;

        // ✅ USAR HELPERS HTTP para evitar bugs
        auto credRes = HttpHelpers::hacerPOST("/credenciales_biometricas", credencial, 15);
        
        json credResponse;
        if (!HttpHelpers::procesarResponsePOST(credRes, credResponse)) {
            std::cerr << "[ERROR] No se pudo registrar credencial en BD" << std::endl;
            response["success"] = false;
            response["error"] = "Error al registrar credencial biométrica";
            if (credRes) {
                response["detail"] = credRes->body;
                response["status"] = credRes->status;
            }
            return response;
        }
        
        std::cout << "[SUCCESS] Credencial registrada exitosamente en BD" << std::endl;
        std::cout << "[DEBUG] Respuesta: " << credResponse.dump() << std::endl;
        std::cout << "[DEBUG] === FIN REGISTRO CREDENCIAL ===" << std::endl;

        response["success"] = true;
        response["message"] = "Biometria de voz registrada exitosamente";
        response["cedula"] = cedula;
        response["user_id"] = nuevoId;
        response["id_usuario_bd"] = idUsuario;
        response["muestras_procesadas"] = featuresList.size();
        response["audios_exitosos"] = audiosExitosos;
        response["audios_fallidos"] = audiosFallidos;

        std::cout << "\n-> Biometria registrada para cedula " << cedula 
                 << " con ID " << nuevoId 
                 << " (" << featuresList.size() << " muestras)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n# ERROR CRITICO: " << e.what() << std::endl;
        response["success"] = false;
        response["error"] = std::string("Error en registro: ") + e.what();
    }

    return response;
}
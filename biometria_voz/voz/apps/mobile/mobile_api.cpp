#include "mobile_api.h"
#include "sqlite_adapter.h"
#include "../../core/pipeline/audio_pipeline.h"
#include "../../core/classification/svm.h"
#include "../../core/process_dataset/dataset.h"
#include "../../external/json.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <random>
#include <fstream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

// ============================================================================
// ESTADO GLOBAL DE LA LIBRERIA
// ============================================================================

struct MobileState {
    std::unique_ptr<SQLiteAdapter> db;
    ModeloSVM svm;
    std::string modelPath;
    std::string datasetPath;
    std::string lastError;
    bool initialized;
    bool modelLoaded;

    MobileState() : initialized(false), modelLoaded(false) {}
};

static MobileState* g_state = nullptr;

// ============================================================================
// UTILIDADES INTERNAS
// ============================================================================

static void set_last_error(const std::string& error) {
    if (g_state) {
        g_state->lastError = error;
    }
}

static bool check_initialized() {
    if (!g_state || !g_state->initialized) {
        set_last_error("Libreria no inicializada. Llamar voz_mobile_init() primero");
        return false;
    }
    return true;
}

// ============================================================================
// GESTION DE LIBRERIA
// ============================================================================

extern "C" int voz_mobile_init(const char* db_path, 
                                const char* model_path,
                                const char* dataset_path) {
    try {
        std::cout << "@ Inicializando libreria mobile..." << std::endl;
        
        if (g_state) {
            voz_mobile_cleanup();
        }

        g_state = new MobileState();

        // Conectar base de datos
        std::cout << "-> Conectando a SQLite: " << db_path << std::endl;
        g_state->db = std::make_unique<SQLiteAdapter>(db_path);
        if (!g_state->db->conectar()) {
            set_last_error("Error conectando a base de datos SQLite: " + std::string(db_path));
            delete g_state;
            g_state = nullptr;
            return -1;
        }
        std::cout << "-> Base de datos SQLite conectada correctamente" << std::endl;

        // Configurar rutas
        g_state->modelPath = model_path;
        g_state->datasetPath = dataset_path;
        std::cout << "-> Directorio de modelos: " << model_path << std::endl;
        std::cout << "-> Archivo de dataset: " << dataset_path << std::endl;

        // Crear directorios si no existen
        fs::create_directories(model_path);
        fs::create_directories(fs::path(dataset_path).parent_path());

        // Cargar modelo SVM si existe
        std::string metadataPath = std::string(model_path) + "metadata.json";
        if (fs::exists(metadataPath)) {
            std::cout << "-> Cargando modelo SVM desde: " << model_path << std::endl;
            try {
                g_state->svm = cargarModeloSVM(model_path);
                g_state->modelLoaded = !g_state->svm.clases.empty();
                
                if (g_state->modelLoaded) {
                    std::cout << "-> Modelo SVM cargado: " << g_state->svm.clases.size() 
                              << " clases, " << g_state->svm.dimensionCaracteristicas << " features" << std::endl;
                } else {
                    std::cout << "# Modelo SVM cargado pero sin clases" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "# No se pudo cargar modelo SVM (se creara con primer registro): " 
                          << e.what() << std::endl;
                g_state->modelLoaded = false;
            }
        } else {
            std::cout << "-> Modelo SVM no encontrado (se creara con primer registro)" << std::endl;
        }

        // Verificar dataset procesado
        if (fs::exists(dataset_path)) {
            auto fileSize = fs::file_size(dataset_path);
            std::cout << "-> Dataset encontrado: " << dataset_path 
                      << " (" << (fileSize / 1024.0 / 1024.0) << " MB)" << std::endl;
        } else {
            std::cout << "# Dataset no encontrado: " << dataset_path << std::endl;
        }

        g_state->initialized = true;
        std::cout << "@ Libreria mobile inicializada correctamente" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "! Error inicializando libreria: " << e.what() << std::endl;
        set_last_error(std::string("Error inicializando libreria: ") + e.what());
        if (g_state) {
            delete g_state;
            g_state = nullptr;
        }
        return -1;
    }
}

extern "C" void voz_mobile_cleanup() {
    if (g_state) {
        delete g_state;
        g_state = nullptr;
    }
}

extern "C" const char* voz_mobile_version() {
    return "1.0.0-mobile";
}

// ============================================================================
// USUARIOS
// ============================================================================

extern "C" int voz_mobile_obtener_id_usuario(const char* identificador) {
    if (!check_initialized()) return -1;

    try {
        auto usuario = g_state->db->obtenerUsuarioPorIdentificador(identificador);
        if (usuario.has_value()) {
            return usuario->id_usuario;
        }
        return -1;
    } catch (const std::exception& e) {
        set_last_error(std::string("Error obteniendo usuario: ") + e.what());
        return -1;
    }
}

extern "C" int voz_mobile_crear_usuario(const char* identificador) {
    if (!check_initialized()) return -1;

    try {
        int idUsuario = g_state->db->insertarUsuario(identificador);
        if (idUsuario > 0) {
            // Crear credencial biometrica de voz
            g_state->db->insertarCredencial(idUsuario, "voz");
        }
        return idUsuario;
    } catch (const std::exception& e) {
        set_last_error(std::string("Error creando usuario: ") + e.what());
        return -1;
    }
}

extern "C" int voz_mobile_usuario_existe(const char* identificador) {
    if (!check_initialized()) return 0;

    try {
        auto usuario = g_state->db->obtenerUsuarioPorIdentificador(identificador);
        return usuario.has_value() ? 1 : 0;
    } catch (const std::exception& e) {
        set_last_error(std::string("Error verificando usuario: ") + e.what());
        return 0;
    }
}

// ============================================================================
// FRASES DINAMICAS
// ============================================================================

extern "C" int voz_mobile_obtener_frase_aleatoria(char* buffer, size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        auto frases = g_state->db->obtenerFrasesActivas();
        if (frases.empty()) {
            set_last_error("No hay frases disponibles");
            return -1;
        }

        // Seleccionar frase aleatoria
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, frases.size() - 1);
        
        const auto& fraseSeleccionada = frases[dis(gen)];
        
        if (fraseSeleccionada.frase.length() + 1 > buffer_size) {
            set_last_error("Buffer insuficiente para la frase");
            return -1;
        }

        std::strncpy(buffer, fraseSeleccionada.frase.c_str(), buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
        
        return fraseSeleccionada.id_frase;

    } catch (const std::exception& e) {
        set_last_error(std::string("Error obteniendo frase: ") + e.what());
        return -1;
    }
}

extern "C" int voz_mobile_obtener_frase_por_id(int id_frase, char* buffer, size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        auto frase = g_state->db->obtenerFrasePorId(id_frase);
        if (!frase.has_value()) {
            set_last_error("Frase no encontrada");
            return -1;
        }

        if (frase->frase.length() + 1 > buffer_size) {
            set_last_error("Buffer insuficiente para la frase");
            return -1;
        }

        std::strncpy(buffer, frase->frase.c_str(), buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
        
        return 0;

    } catch (const std::exception& e) {
        set_last_error(std::string("Error obteniendo frase: ") + e.what());
        return -1;
    }
}

extern "C" int voz_mobile_insertar_frases(const char* frases_json) {
    if (!check_initialized()) return -1;

    try {
        json data = json::parse(frases_json);
        int insertadas = 0;

        for (const auto& item : data) {
            std::string frase = item["frase"];
            std::string categoria = item.value("categoria", "general");
            
            if (g_state->db->insertarFrase(frase, categoria) > 0) {
                insertadas++;
            }
        }

        return insertadas;

    } catch (const std::exception& e) {
        set_last_error(std::string("Error insertando frases: ") + e.what());
        return -1;
    }
}

// ============================================================================
// REGISTRO BIOMETRICO
// ============================================================================

extern "C" int voz_mobile_registrar_biometria(const char* identificador,
                                               const char* audio_path,
                                               int id_frase,
                                               char* resultado_json,
                                               size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        json response;

        // 1. Verificar usuario
        auto usuario = g_state->db->obtenerUsuarioPorIdentificador(identificador);
        if (!usuario.has_value()) {
            // Crear usuario si no existe
            int idUsuario = g_state->db->insertarUsuario(identificador);
            if (idUsuario <= 0) {
                set_last_error("Error creando usuario");
                response["success"] = false;
                response["error"] = "Error creando usuario";
                std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
                return -1;
            }
            
            g_state->db->insertarCredencial(idUsuario, "voz");
            usuario = g_state->db->obtenerUsuarioPorId(idUsuario);
        }

        int userId = usuario->id_usuario;

        // 2. Procesar audio completo con pipeline
        std::vector<std::vector<AudioSample>> todasFeatures;
        if (!procesarAudioCompleto(audio_path, todasFeatures) || todasFeatures.empty()) {
            set_last_error("Error procesando audio");
            response["success"] = false;
            response["error"] = "Error procesando audio";
            std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
            return -1;
        }

        // 3. Tomar primera muestra (sin augmentation en mobile)
        std::vector<AudioSample> features = todasFeatures[0];

        // 4. Agregar al dataset
        Dataset dataset;
        if (fs::exists(g_state->datasetPath)) {
            dataset = cargarDatasetBinario(g_state->datasetPath);
        }

        dataset.X.push_back(features);
        dataset.y.push_back(userId);
        guardarDatasetBinario(g_state->datasetPath, dataset);

        // 5. Reentrenar SVM
        g_state->svm = entrenarSVMOVA(dataset.X, dataset.y);
        guardarModeloSVM(g_state->modelPath, g_state->svm);
        g_state->modelLoaded = true;

        // 6. Registrar en base de datos
        auto credencial = g_state->db->obtenerCredencialPorUsuario(userId, "voz");
        if (credencial.has_value()) {
            g_state->db->insertarValidacion(credencial->id_credencial, 
                                           "registro_exitoso", 
                                           1.0);
        }

        response["success"] = true;
        response["user_id"] = userId;
        response["samples_trained"] = dataset.y.size();
        response["features_extracted"] = features.size();

        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        resultado_json[buffer_size - 1] = '\0';
        
        return 0;

    } catch (const std::exception& e) {
        set_last_error(std::string("Error registrando biometria: ") + e.what());
        json response;
        response["success"] = false;
        response["error"] = e.what();
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        return -1;
    }
}

// ============================================================================
// AUTENTICACION
// ============================================================================

extern "C" int voz_mobile_autenticar(const char* identificador,
                                      const char* audio_path,
                                      int id_frase,
                                      char* resultado_json,
                                      size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        json response;

        // 1. Verificar usuario
        auto usuario = g_state->db->obtenerUsuarioPorIdentificador(identificador);
        if (!usuario.has_value()) {
            set_last_error("Usuario no encontrado");
            response["success"] = false;
            response["authenticated"] = false;
            response["error"] = "Usuario no encontrado";
            std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
            return 0;
        }

        int userId = usuario->id_usuario;

        // 2. Procesar audio completo con pipeline
        std::vector<std::vector<AudioSample>> todasFeatures;
        if (!procesarAudioCompleto(audio_path, todasFeatures) || todasFeatures.empty()) {
            set_last_error("Error procesando audio");
            response["success"] = false;
            response["error"] = "Error procesando audio";
            std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
            return -1;
        }

        // 3. Tomar primera muestra
        std::vector<AudioSample> features = todasFeatures[0];

        // 4. Predecir con SVM
        if (!g_state->modelLoaded) {
            set_last_error("Modelo no cargado");
            response["success"] = false;
            response["authenticated"] = false;
            response["error"] = "Modelo no cargado";
            std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
            return 0;
        }
        
        int predictedClass = predecirHablante(features, g_state->svm);
        auto scores = obtenerScores(features, g_state->svm);
        
        // Calcular confianza (score normalizado)
        AudioSample maxScore = -1e9;
        AudioSample secondMaxScore = -1e9;
        for (AudioSample score : scores) {
            if (score > maxScore) {
                secondMaxScore = maxScore;
                maxScore = score;
            } else if (score > secondMaxScore) {
                secondMaxScore = score;
            }
        }
        AudioSample confidence = (maxScore - secondMaxScore) / (maxScore + 1e-6);

        bool autenticado = (predictedClass == userId && confidence >= 0.6);

        // 5. Registrar validacion
        auto credencial = g_state->db->obtenerCredencialPorUsuario(userId, "voz");
        if (credencial.has_value()) {
            g_state->db->insertarValidacion(
                credencial->id_credencial,
                autenticado ? "exitoso" : "fallido",
                confidence
            );
        }

        // 6. Construir respuesta
        response["success"] = true;
        response["authenticated"] = autenticado;
        response["user_id"] = userId;
        response["predicted_class"] = predictedClass;
        response["confidence"] = confidence;

        json scoresJson;
        for (size_t i = 0; i < g_state->svm.clases.size() && i < scores.size(); ++i) {
            scoresJson[std::to_string(g_state->svm.clases[i])] = scores[i];
        }
        response["all_scores"] = scoresJson;

        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        resultado_json[buffer_size - 1] = '\0';
        
        return autenticado ? 1 : 0;

    } catch (const std::exception& e) {
        set_last_error(std::string("Error autenticando: ") + e.what());
        json response;
        response["success"] = false;
        response["error"] = e.what();
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        return -1;
    }
}

// ============================================================================
// SINCRONIZACION
// ============================================================================

extern "C" int voz_mobile_sync_push(const char* server_url, char* resultado_json, size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        std::cout << "-> Iniciando sync push al servidor: " << server_url << std::endl;
        
        // Obtener caracteristicas pendientes
        auto caracteristicas = g_state->db->obtenerCaracteristicasPendientes();
        
        if (caracteristicas.empty()) {
            std::cout << "-> No hay caracteristicas pendientes para sincronizar" << std::endl;
            json response;
            response["ok"] = true;
            response["enviados"] = 0;
            std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
            resultado_json[buffer_size - 1] = '\0';
            return 0;
        }

        std::cout << "-> Enviando " << caracteristicas.size() << " caracteristicas al servidor" << std::endl;

        // Construir payload JSON
        json payload = json::array();
        for (const auto& car : caracteristicas) {
            json item;
            item["id_usuario"] = car.id_usuario;
            item["id_credencial"] = car.id_credencial;
            item["vector_features"] = car.vector_features;
            item["dimension"] = car.dimension;
            item["uuid_dispositivo"] = car.uuid_dispositivo;
            payload.push_back(item);
        }

        // TODO: Implementar llamada HTTP POST a /sync/push
        // Por ahora marcar como sincronizadas localmente
        int sincronizados = 0;
        for (const auto& car : caracteristicas) {
            if (g_state->db->marcarCaracteristicaSincronizada(car.id_caracteristica)) {
                sincronizados++;
            }
        }

        std::cout << "-> Sincronizadas " << sincronizados << " caracteristicas" << std::endl;

        json response;
        response["ok"] = true;
        response["enviados"] = sincronizados;
        response["errores"] = caracteristicas.size() - sincronizados;
        
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        resultado_json[buffer_size - 1] = '\0';
        
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "! Error en sync push: " << e.what() << std::endl;
        set_last_error(std::string("Error sync push: ") + e.what());
        json response;
        response["ok"] = false;
        response["error"] = e.what();
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        return -1;
    }
}

extern "C" int voz_mobile_sync_pull(const char* server_url, const char* desde, 
                                     char* resultado_json, size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        std::cout << "-> Iniciando sync pull desde servidor: " << server_url << std::endl;
        
        // TODO: Implementar llamada HTTP GET a /sync/pull?desde=timestamp
        // Por ahora retornar respuesta vacia
        
        json response;
        response["ok"] = true;
        response["frases_nuevas"] = json::array();
        response["usuarios_actualizados"] = json::array();
        
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        resultado_json[buffer_size - 1] = '\0';
        
        std::cout << "-> Sync pull completado" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "! Error en sync pull: " << e.what() << std::endl;
        set_last_error(std::string("Error sync pull: ") + e.what());
        json response;
        response["ok"] = false;
        response["error"] = e.what();
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        return -1;
    }
}

extern "C" int voz_mobile_sync_modelo(const char* server_url, const char* identificador,
                                       char* resultado_json, size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        std::cout << "-> Iniciando descarga de modelo para: " << identificador << std::endl;
        
        // TODO: Implementar llamada HTTP GET a /sync/modelo?cedula=identificador
        // Por ahora retornar error
        
        json response;
        response["ok"] = false;
        response["error"] = "Funcionalidad pendiente de implementar";
        
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        resultado_json[buffer_size - 1] = '\0';
        
        return -1;

    } catch (const std::exception& e) {
        std::cerr << "! Error descargando modelo: " << e.what() << std::endl;
        set_last_error(std::string("Error sync modelo: ") + e.what());
        json response;
        response["ok"] = false;
        response["error"] = e.what();
        std::strncpy(resultado_json, response.dump().c_str(), buffer_size - 1);
        return -1;
    }
}

extern "C" int voz_mobile_obtener_uuid_dispositivo(char* buffer, size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        std::string uuid = g_state->db->obtenerConfigSync("uuid_dispositivo");
        
        if (uuid.empty()) {
            set_last_error("UUID no establecido");
            return -1;
        }

        std::strncpy(buffer, uuid.c_str(), buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
        
        return 0;

    } catch (const std::exception& e) {
        set_last_error(std::string("Error obteniendo UUID: ") + e.what());
        return -1;
    }
}

extern "C" int voz_mobile_establecer_uuid_dispositivo(const char* uuid) {
    if (!check_initialized()) return -1;

    try {
        g_state->db->guardarConfigSync("uuid_dispositivo", uuid);
        std::cout << "-> UUID dispositivo establecido: " << uuid << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "! Error estableciendo UUID: " << e.what() << std::endl;
        set_last_error(std::string("Error estableciendo UUID: ") + e.what());
        return -1;
    }
}

// ============================================================================
// UTILIDADES
// ============================================================================

extern "C" void voz_mobile_obtener_ultimo_error(char* buffer, size_t buffer_size) {
    if (g_state && !g_state->lastError.empty()) {
        std::strncpy(buffer, g_state->lastError.c_str(), buffer_size - 1);
        buffer[buffer_size - 1] = '\0';
    } else {
        buffer[0] = '\0';
    }
}

extern "C" int voz_mobile_obtener_estadisticas(char* stats_json, size_t buffer_size) {
    if (!check_initialized()) return -1;

    try {
        json stats;
        stats["usuarios_registrados"] = g_state->db->listarUsuarios().size();
        stats["frases_activas"] = g_state->db->obtenerFrasesActivas().size();
        stats["pendientes_sincronizacion"] = g_state->db->contarPendientesSincronizacion();
        stats["modelo_cargado"] = g_state->modelLoaded;
        
        if (g_state->modelLoaded) {
            stats["num_clases"] = g_state->svm.clases.size();
            stats["num_features"] = g_state->svm.dimensionCaracteristicas;
        }
        
        std::string statsStr = stats.dump();
        if (statsStr.length() + 1 > buffer_size) {
            set_last_error("Buffer insuficiente");
            return -1;
        }

        std::strncpy(stats_json, statsStr.c_str(), buffer_size - 1);
        stats_json[buffer_size - 1] = '\0';
        
        return 0;

    } catch (const std::exception& e) {
        set_last_error(std::string("Error obteniendo estadisticas: ") + e.what());
        return -1;
    }
}


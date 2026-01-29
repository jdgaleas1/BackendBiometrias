#include <utility>
#include "autenticacion_service.h"
#include "../../utils/http_helpers.h"
#include "../../external/httplib.h"
#include "../../external/json.hpp"
#include "../../core/pipeline/audio_pipeline.h"
#include "../../core/classification/svm.h"
#include "../../core/asr/similaridad.h"
#include "../../core/asr/whisper_asr.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iostream>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Normalización está en core/classification/svm_utils.cpp

AutenticacionService::AutenticacionService(const std::string& modelPath,
    const std::string& mappingPath) {
    recargarModelo(modelPath);
    recargarMapeos(mappingPath);  // mappingPath ahora se ignora, usa metadata.json
}

void AutenticacionService::recargarModelo(const std::string& modelPath) {
    // Cargar modelo modular si es un directorio
    if (fs::is_directory(modelPath)) {
        // Cargar modelo modular desde directorio
        modelo = cargarModeloModular(modelPath);
    } else if (fs::exists(modelPath)) {
        // Fallback: cargar modelo monolítico antiguo (compatibilidad)
        modelo = cargarModeloSVM(modelPath);
        std::cout << "-> Modelo SVM monolítico cargado: " << modelo.clases.size() << " clases" << std::endl;
    } else {
        std::cout << "! Modelo no encontrado: " << modelPath << std::endl;
    }
}

void AutenticacionService::recargarMapeos(const std::string& mappingPath) {
    mapeoUsuarios.clear();

    // Cargar desde metadata.json (fuente de verdad)
    std::string metadataPath = obtenerRutaModelo() + "metadata.json";
    std::ifstream metaIn(metadataPath);
    
    if (metaIn.is_open()) {
        try {
            json metadata;
            metaIn >> metadata;
            metaIn.close();

            // Extraer lista de clases activas
            auto clases = metadata["classes"];
            
            // Por ahora usamos el ID como nombre (hasta tener DB real)
            for (int userId : clases) {
                mapeoUsuarios[userId] = "Usuario_" + std::to_string(userId);
            }

        } catch (const std::exception& e) {
            std::cerr << "! Error parseando metadata.json: " << e.what() << std::endl;
        }
    } else {
        // Fallback antiguo: intentar leer speaker_mapping.txt si existe
        std::ifstream mapeo(mappingPath);
        if (mapeo.is_open()) {
            std::string linea;
            while (std::getline(mapeo, linea)) {
                if (linea.empty() || linea[0] == '#') continue;

                std::istringstream iss(linea);
                int id;
                std::string nombre;
                if (iss >> id >> nombre) {
                    mapeoUsuarios[id] = nombre;
                }
            }
            mapeo.close();
        }
    }
}

bool AutenticacionService::procesarAudio(const std::string& audioPath, std::vector<AudioSample>& features) {
    features.clear();

    // USAR PIPELINE CENTRALIZADO
    std::vector<std::vector<AudioSample>> todasFeatures;
    if (!procesarAudioCompleto(audioPath, todasFeatures) || todasFeatures.empty()) {
        return false;
    }

    // Tomar solo el primero (sin augmentation en autenticación)
    features = todasFeatures[0];
    
    // ✅ VALIDAR dimensión considerando expansión polinomial
    int expectedFeatures = CONFIG_MFCC.totalFeatures;
    if (CONFIG_SVM.usarExpansionPolinomial) {
        expectedFeatures *= 2;  // Se duplica con la expansión
    }
    
    if (features.size() != static_cast<size_t>(expectedFeatures)) {
        std::cerr << "! Error: Features extraidos (" << features.size() 
                  << ") no coinciden con dimension esperada (" 
                  << expectedFeatures << ")" << std::endl;
        if (CONFIG_SVM.usarExpansionPolinomial) {
            std::cerr << "   (" << CONFIG_MFCC.totalFeatures 
                      << " base + expansion polinomial = " << expectedFeatures << ")" << std::endl;
        }
        return false;
    }
    
    std::cout << "   * Features extraidos: " << features.size();
    if (CONFIG_SVM.usarExpansionPolinomial) {
        std::cout << " (" << CONFIG_MFCC.totalFeatures << " base + expansion polinomial)";
    }
    std::cout << std::endl;
    
    return true;
}
ResultadoAutenticacion AutenticacionService::autenticar(const std::string& audioPath, const std::string& identificador, int idFrase) {
    ResultadoAutenticacion resultado;
    auto inicio = std::chrono::high_resolution_clock::now();

    try {
        // 1. Validar que el identificador existe en la base de datos
        if (!identificador.empty()) {
            std::cout << "\n[DEBUG] === VALIDACION IDENTIFICADOR ===" << std::endl;
            std::cout << "[DEBUG] Identificador a validar: " << identificador << std::endl;
            
            std::string endpoint = "/usuarios?identificador_unico=eq." + identificador;
            std::cout << "[DEBUG] Endpoint GET: " << endpoint << std::endl;
            
            auto res = HttpHelpers::hacerGET(endpoint, 15);
            
            if (!res) {
                std::cerr << "[ERROR] Response nullptr - No conexion a BD" << std::endl;
                resultado.exito = false;
                resultado.error = "No se pudo conectar a la base de datos";
                return resultado;
            }
            
            std::cout << "[DEBUG] Response Status: " << res->status << std::endl;
            
            if (res->status != 200) {
                std::cerr << "[ERROR] Status != 200, identificador no existe" << std::endl;
                resultado.exito = false;
                resultado.error = "Identificador no registrado en el sistema";
                return resultado;
            }
            
            try {
                auto usuarios = json::parse(res->body);
                std::cout << "[DEBUG] Usuarios encontrados: " << usuarios.size() << std::endl;
                
                if (usuarios.empty()) {
                    std::cerr << "[ERROR] Array vacio, identificador no existe" << std::endl;
                    resultado.exito = false;
                    resultado.error = "Identificador no registrado en el sistema";
                    return resultado;
                }
                
                std::cout << "[DEBUG] Identificador validado OK" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Error parsing JSON: " << e.what() << std::endl;
                resultado.exito = false;
                resultado.error = "Error validando identificador";
                return resultado;
            }
            
            std::cout << "[DEBUG] === FIN VALIDACION IDENTIFICADOR ===" << std::endl;
        }
        
        // 2. Verificar modelo
        if (modelo.clases.empty()) {
            resultado.exito = false;
            resultado.error = "No hay modelo entrenado";
            return resultado;
        }

        // Procesar audio
        std::vector<AudioSample> features;
        if (!procesarAudio(audioPath, features)) {
            resultado.exito = false;
            resultado.error = "Error procesando audio";
            return resultado;
        }
        
        int idPredecido = predecirHablante(features, modelo);

        // Obtener scores
        auto scores = obtenerScores(features, modelo);

        // ========================================================================
        // DEBUGGING: Imprimir TOP 5 scores para identificar confusiones
        // ========================================================================
        std::vector<std::pair<int, double>> scoresConClase;
        for (size_t i = 0; i < modelo.clases.size(); ++i) {
            scoresConClase.push_back({modelo.clases[i], scores[i]});
        }
        
        // Ordenar por score descendente
        std::sort(scoresConClase.begin(), scoresConClase.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "\n  [DEBUG] TOP 5 SCORES:" << std::endl;
        for (int i = 0; i < std::min(5, static_cast<int>(scoresConClase.size())); ++i) {
            std::cout << "     #" << (i+1) << ": Clase " << scoresConClase[i].first 
                      << " | Score: " << std::fixed << std::setprecision(4) 
                      << scoresConClase[i].second;
            if (i == 0) std::cout << " <- PREDICHO";
            std::cout << std::endl;
        }

        // ✅ CONTROL DE ACCESO CORREGIDO (sin duplicados)
        auto scoresCopia = scores;
        std::sort(scoresCopia.begin(), scoresCopia.end(), std::greater<double>());

        double maxScore = scoresCopia[0];
        double segundoScore = scoresCopia.size() > 1 ? scoresCopia[1] : -999.0;
        double tercerScore = scoresCopia.size() > 2 ? scoresCopia[2] : -999.0;
        double diferencia = maxScore - segundoScore;

        // UMBRALES DESDE CONFIG (config.h)
        // Ajustados para 250 features que generan mejores scores
        const double SCORE_MINIMO = CONFIG_AUTH.scoreMinimo;
        const double DIFERENCIA_MINIMA = CONFIG_AUTH.diferenciaMinima;
        const double FACTOR_SEGUNDO = CONFIG_AUTH.factorSegundoLugar;
        const double UMBRAL_ALTO = CONFIG_AUTH.umbralScoreAlto;

        std::cout << "  [DEBUG] Scores: max=" << maxScore 
                  << " | segundo=" << segundoScore 
                  << " | diferencia=" << diferencia << std::endl;
        std::cout << "  [DEBUG] Umbrales: scoreMin=" << SCORE_MINIMO
                  << " | difMin=" << DIFERENCIA_MINIMA
                  << " | factorSeg=" << FACTOR_SEGUNDO << std::endl;

        // LOGICA DE DECISION MEJORADA
        // Sistema adaptativo: permite autenticar con diferentes niveles de evidencia
        
        bool scoreAltoSuficiente = (maxScore >= SCORE_MINIMO);
        bool tieneSeparacionClara = (diferencia >= DIFERENCIA_MINIMA);
        bool segundoLugarBajo = (segundoScore < (maxScore * FACTOR_SEGUNDO));
        
        std::cout << "  [DEBUG] Condiciones: scoreAlto=" << scoreAltoSuficiente 
                  << " | separacion=" << tieneSeparacionClara 
                  << " | segundoBajo=" << segundoLugarBajo << std::endl;

        // DECISION FINAL - Logica adaptativa con multiples criterios
        // Se autentica si cumple CUALQUIERA de estos escenarios:
        // 
        // ESCENARIO 1: Todas las condiciones (ideal - maxima confianza)
        bool todasCondiciones = scoreAltoSuficiente && tieneSeparacionClara && segundoLugarBajo;
        
        // ESCENARIO 2: Score muy alto, incluso sin mucha separacion
        bool scoreExcelente = (maxScore >= UMBRAL_ALTO);
        
        // ESCENARIO 3: Score aceptable + buena separacion (al menos una de las dos condiciones extra)
        bool scoreYSeparacion = scoreAltoSuficiente && (tieneSeparacionClara || segundoLugarBajo);
        
        bool autenticado = todasCondiciones || scoreExcelente || scoreYSeparacion;
        
        std::cout << "  [DEBUG] Escenarios: todas=" << todasCondiciones 
                  << " | excelente=" << scoreExcelente 
                  << " | combinado=" << scoreYSeparacion 
                  << " -> FINAL=" << autenticado << std::endl;

        // CALCULO DE CONFIANZA - Basado en score y separacion
        double confianza;
        
        if (!autenticado) {
            // Si no esta autenticado, confianza baja
            confianza = std::min(0.40, maxScore / SCORE_MINIMO);
            confianza = std::max(0.0, confianza);
        } else {
            // Autenticado: calcular confianza segun score y separacion
            if (maxScore >= UMBRAL_ALTO) {
                // Score muy alto: confianza 95-100%
                confianza = 0.95 + std::min(0.05, (maxScore - UMBRAL_ALTO) * 0.02);
            } else if (maxScore >= SCORE_MINIMO) {
                // Score aceptable: confianza 70-95%
                double rango = UMBRAL_ALTO - SCORE_MINIMO;
                double posicion = maxScore - SCORE_MINIMO;
                confianza = 0.70 + (posicion / rango) * 0.25;
            } else {
                // No deberia llegar aqui si autenticado=true
                confianza = 0.50;
            }
            
            // Bonus por separacion muy alta
            if (diferencia > 2.0) {
                confianza = std::min(1.0, confianza * 1.05);
            }
        }

        // Construir resultado
        resultado.exito = true;
        resultado.userId = idPredecido;
        resultado.userName = mapeoUsuarios.count(idPredecido) ?
            mapeoUsuarios[idPredecido] : "Desconocido";
        resultado.confianza = confianza;
        resultado.autenticado = autenticado;

        std::cout << "-> Auth: ID " << idPredecido << " - " 
                  << (autenticado ? "AUTORIZADO" : "DENEGADO") 
                  << " (conf: " << confianza << ")" << std::endl;

        // Guardar scores
        for (size_t i = 0; i < modelo.clases.size(); ++i) {
            resultado.scores[modelo.clases[i]] = scores[i];
        }

        auto fin = std::chrono::high_resolution_clock::now();
        resultado.tiempoProcesamiento =
            std::chrono::duration_cast<std::chrono::milliseconds>(fin - inicio).count();

    }
    catch (const std::exception& e) {
        resultado.exito = false;
        resultado.error = std::string("Excepcion: ") + e.what();
    }
    if (resultado.autenticado && idFrase > 0) {
        std::cout << "\n-> VERIFICACION DE FRASE DINAMICA" << std::endl;
        std::cout << "   ID Frase solicitada: " << idFrase << std::endl;

        // Usar directamente el servicio en lugar de HTTP interno
        auto fraseResult = frasesService.obtenerFrasePorId(idFrase);

        if (fraseResult.contains("frase")) {
            resultado.fraseEsperada = fraseResult["frase"];
            std::cout << "   * Frase esperada: \"" << resultado.fraseEsperada << "\"" << std::endl;

            // OBTENER TRANSCRIPCIÓN
            resultado.transcripcionDetectada = obtenerTranscripcion(audioPath);
            std::cout << "   * Transcripcion detectada: \"" << resultado.transcripcionDetectada << "\"" << std::endl;

            // Calcular similitud
            std::string fraseNormalizada = normalizarTxt(resultado.fraseEsperada);
            std::string transcripcionNormalizada = normalizarTxt(resultado.transcripcionDetectada);
            
            resultado.similitudTexto = porcentajeSimilitud(fraseNormalizada, transcripcionNormalizada);
            
            std::cout << "   @ Frase normalizada: \"" << fraseNormalizada << "\"" << std::endl;
            std::cout << "   @ Transcripcion normalizada: \"" << transcripcionNormalizada << "\"" << std::endl;
            std::cout << "   @ Similitud calculada: " << (resultado.similitudTexto * 100.0) << "%" << std::endl;

            // UMBRAL AJUSTADO: 70% (antes era 85%, demasiado estricto)
            const double UMBRAL_SIMILITUD = 0.70;
            resultado.textoCoincide = (resultado.similitudTexto >= UMBRAL_SIMILITUD);
            
            // Verificación doble: identidad + texto correcto
            resultado.autenticado = resultado.autenticado && resultado.textoCoincide;

            std::cout << "   @ Umbral requerido: " << (UMBRAL_SIMILITUD * 100.0) << "%" << std::endl;
            std::cout << "   @ Texto coincide: " << (resultado.textoCoincide ? "SI" : "NO") << std::endl;
            std::cout << "   @ Autenticacion final: " << (resultado.autenticado ? "AUTORIZADO" : "DENEGADO") << std::endl;
        } else {
            std::cout << "   ! ERROR: No se pudo obtener frase con ID " << idFrase << std::endl;
        }
    }
    return resultado;
}
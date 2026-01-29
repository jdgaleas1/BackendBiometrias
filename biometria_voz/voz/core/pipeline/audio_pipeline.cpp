#include "audio_pipeline.h"
#include "../../utils/config.h"
#include "../load_audio/audio_io.h"
#include "../preprocessing/preprocesar.h"
#include "../segmentation/stft.h"
#include "../features/mfcc.h"
#include "../augmentation/audio_augmentation.h"
#include <iostream>
#include <optional>
#include <cmath>

// FUNCION INTERNA (PRIVADA): Procesa UN buffer 
static std::optional<std::vector<AudioSample>> procesarUnBuffer(
    const std::vector<AudioSample>& audioBuffer,
    int sampleRate)
{
    // Validar duracion minima
    if (audioBuffer.size() < static_cast<size_t>(CONFIG_DATASET.minAudioSamples)) {
        return std::nullopt;
    }

    if (CONFIG_PREP.verbose) {
        std::cout << "\n========== PIPELINE/FLUJO AUDIO ==========" << std::endl;
        CONFIG_PREP.mostrar();
        std::cout << std::string(50, '-') << std::endl;
    }

    try {
        // Declarar variable para audio procesado
        std::vector<AudioSample> procesadoAudio;

        // BYPASS OPCIONAL DE PREPROCESAMIENTO (PARA PRUEBAS)
        if (!CONFIG_PREP.enablePreprocessing) {
            if (CONFIG_PREP.verbose) {
                std::cout << "\n*** BYPASS: Saltando preprocesamiento completo ***" << std::endl;
                std::cout << "-> Audio pasa directo a STFT" << std::endl;
            }
            procesadoAudio = audioBuffer;
        }
        else {
            // PASO 1: NORMALIZACION RMS
            // Estabiliza amplitud para etapas posteriores
            auto normalized = normalizeRMS(audioBuffer, CONFIG_PREP.normalizationTargetRMS);
            if (normalized.empty()) {
                std::cerr << "! Pipeline: Fallo en normalizacion" << std::endl;
                return std::nullopt;
            }

            // PASO 2: VAD 
            auto voz = applyVAD(normalized, sampleRate);
            if (voz.empty() ||
                voz.size() < static_cast<size_t>(CONFIG_DATASET.minAudioSamples)) {
                std::cerr << "! Pipeline: VAD no detecto suficiente voz" << std::endl;
                return std::nullopt;
            }

            procesadoAudio = std::move(voz);
        }

        // PASO 3: STFT
        // Transformada tiempo-frecuencia con precision double
        auto stft = applySTFT(procesadoAudio, sampleRate);
        if (stft.empty()) {
            std::cerr << "! Pipeline: Fallo en STFT" << std::endl;
            return std::nullopt;
        }

        // PASO 4: MFCC
        // Extraccion de coeficientes cepstrales
        auto mfcc = extractMFCC(stft, sampleRate);
        if (mfcc.empty()) {
            std::cerr << "! Pipeline: Fallo en MFCC" << std::endl;
            return std::nullopt;
        }

        // PASO 5: ESTADISTICAS
        // Calcular mean de MFCCs
        auto features = calcularEstadisticasMFCC(mfcc);

        // Validar dimension
        if (features.size() != static_cast<size_t>(CONFIG_MFCC.totalFeatures)) {
            std::cerr << "! Pipeline: Dimension de features incorrecta" << std::endl;
            return std::nullopt;
        }

        // PASO 6: EXPANSION POLINOMIAL (si esta habilitada)
        // IMPORTANTE: Hacer ANTES de normalización L2
        // Transforma: [x1...xN] -> [x1...xN, x1²...xN²]
        // Permite fronteras de decision cuadraticas con SVM lineal
        if (CONFIG_SVM.usarExpansionPolinomial) {
            size_t n_original = features.size();
            std::vector<AudioSample> cuadraticas(n_original);
            
            // Calcular terminos cuadraticos
            for (size_t i = 0; i < n_original; ++i) {
                cuadraticas[i] = features[i] * features[i];
            }
            
            // Concatenar: features = [originales, cuadraticas]
            features.insert(features.end(), cuadraticas.begin(), cuadraticas.end());
            
            if (CONFIG_PREP.verbose) {
                std::cout << "-> Expansion polinomial aplicada (" << n_original 
                          << " -> " << features.size() << " features)" << std::endl;
            }
        }

        // PASO 7: NORMALIZACION L2 (si esta habilitada)
        // Normaliza el vector completo [originales + cuadraticas] a norma unitaria
        if (CONFIG_SVM.usarNormalizacionL2) {
            AudioSample norma = 0.0;
            for (AudioSample val : features) {
                norma += val * val;
            }
            norma = std::sqrt(norma);

            if (norma > 1e-10) {
                for (AudioSample& val : features) {
                    val /= norma;
                }
            }
            else if (CONFIG_PREP.verbose) {
                std::cerr << "% Warning: Norma ~0, no normalizado" << std::endl;
            }
        } else if (CONFIG_PREP.verbose) {
            std::cout << "-> Normalizacion L2: DESACTIVADA" << std::endl;
        }

        if (CONFIG_PREP.verbose) {
            std::cout << "[PROCESAMIENTO COMPLETO]" << std::endl;
            std::cout << "   * Normalizacion -> VAD -> STFT -> MFCC -> Stats" << std::endl;
        }

        return features;
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "! Pipeline: Error de memoria: " << e.what() << std::endl;
        return std::nullopt;
    }
    catch (const std::exception& e) {
        std::cerr << "! Pipeline: Error: " << e.what() << std::endl;
        return std::nullopt;
    }
}
// FUNCION PRINCIPAL: Maneja augmentation y procesa archivo completo
bool procesarAudioCompleto(
    const std::filesystem::path& audioPath,
    std::vector<std::vector<AudioSample>>& outFeatures)
{
    outFeatures.clear();
    // PASO 1: Cargar audio desde archivo 
    int sr, ch, samples;
    auto audio = loadAudio(audioPath.string().c_str(), sr, ch, samples);

    if (audio.empty()) {
        std::cerr << "! Pipeline: Error al cargar " << audioPath.filename() << std::endl;
        return false;
    }

    if (samples < CONFIG_DATASET.minAudioSamples) {
        std::cerr << "! Pipeline: Audio muy corto" << std::endl;
        return false;
    }

    // PASO 2: Decidir si usar augmentation
    const bool usarAugmentation = CONFIG_DATASET.usarAugmentation &&
        (CONFIG_AUG.numVariaciones > 0);

    if (!usarAugmentation) {
        // MODO SIN AUGMENTATION: Procesar solo el original
        auto resultado = procesarUnBuffer(audio, sr);

        if (resultado.has_value()) {
            outFeatures.push_back(resultado.value());
            return true;
        }
        return false;
    }

    // MODO CON AUGMENTATION: Generar N versiones y procesar cada una
    // Generar variaciones 
    auto variaciones = generarVariacionesAudio(audio, CONFIG_AUG.numVariaciones);

    if (variaciones.empty()) {
        std::cerr << "! Pipeline: Fallo generando variaciones" << std::endl;
        return false;
    }

    // Procesar cada variacion (original + perturbadas)
    int exitosas = 0;

    for (const auto& variacion : variaciones) {
        auto resultado = procesarUnBuffer(variacion, sr);

        if (resultado.has_value()) {
            outFeatures.push_back(resultado.value());
            exitosas++;
        }
    }

    return exitosas > 0;
}
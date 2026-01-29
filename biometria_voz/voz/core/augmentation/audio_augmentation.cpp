#include "audio_augmentation.h"
#include "config.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

// ================================================================================
// GENERADOR DE NUMEROS ALEATORIOS - VERSION MODERNA
// ================================================================================

static std::mt19937& obtenerGenerador(unsigned int seed = 0) {
    static std::mt19937 generador(seed);
    static bool inicializado = false;

    if (!inicializado || seed != 0) {
        generador.seed(seed);
        inicializado = true;
    }

    return generador;
}

// ================================================================================
// UTILIDADES INTERNAS
// ================================================================================

/**
 * Soft clipping con funcion tanh para evitar saturacion
 */
static inline AudioSample softClip(AudioSample x) {
    if (x > 1.0) {
        return 1.0 - 0.1 * (1.0 - std::tanh((x - 1.0) * 2.0));
    }
    else if (x < -1.0) {
        return -1.0 + 0.1 * (1.0 - std::tanh((-x - 1.0) * 2.0));
    }
    return x;
}

// ================================================================================
// PERTURBACIONES INDIVIDUALES - VERSION REFACTORIZADA
// ================================================================================

std::vector<AudioSample> aplicarRuidoBlanco(const std::vector<AudioSample>& audio,
    AudioSample intensidad,
    unsigned int seed) {
    if (audio.empty()) {
        std::cerr << "! Error: Audio vacio en aplicarRuidoBlanco" << std::endl;
        return {};
    }

    // Copiar audio original
    std::vector<AudioSample> resultado = audio;

    // Obtener generador
    auto& gen = obtenerGenerador(seed);

    // Agregar ruido gaussiano
    std::normal_distribution<AudioSample> distribucionRuido(0.0, intensidad);

    for (auto& sample : resultado) {
        sample += distribucionRuido(gen);
        sample = softClip(sample);
    }

    return resultado;
}

std::vector<AudioSample> aplicarEscaladoVolumen(const std::vector<AudioSample>& audio,
    AudioSample factor) {
    if (audio.empty()) {
        std::cerr << "! Error: Audio vacio en aplicarEscaladoVolumen" << std::endl;
        return {};
    }

    std::vector<AudioSample> resultado;
    resultado.reserve(audio.size());

    for (const auto& sample : audio) {
        AudioSample escalado = sample * factor;
        resultado.push_back(softClip(escalado));
    }

    return resultado;
}

std::vector<AudioSample> aplicarCambioVelocidad(const std::vector<AudioSample>& audio,
    AudioSample factor) {
    if (audio.empty()) {
        std::cerr << "! Error: Audio vacio en aplicarCambioVelocidad" << std::endl;
        return {};
    }

    if (factor <= 0.0) {
        std::cerr << "! Error: Factor de velocidad debe ser > 0" << std::endl;
        return {};
    }

    // Calcular nuevo numero de muestras
    int numSamples = static_cast<int>(audio.size());
    int newSamples = static_cast<int>(numSamples / factor);
    if (newSamples < 1) newSamples = 1;

    std::vector<AudioSample> resultado;
    resultado.reserve(newSamples);

    // Resampling con interpolacion lineal
    for (int i = 0; i < newSamples; ++i) {
        AudioSample pos = i * factor;
        int idx = static_cast<int>(pos);
        AudioSample frac = pos - idx;

        if (idx + 1 < numSamples) {
            // Interpolacion lineal entre dos muestras
            AudioSample interpolado = audio[idx] * (1.0 - frac) + audio[idx + 1] * frac;
            resultado.push_back(softClip(interpolado));
        }
        else if (idx < numSamples) {
            // Ultima muestra (sin interpolacion)
            resultado.push_back(softClip(audio[idx]));
        }
        else {
            // Fuera de rango (silencio)
            resultado.push_back(0.0);
        }
    }

    return resultado;
}

// ================================================================================
// APLICAR PERTURBACION INDIVIDUAL
// ================================================================================

std::vector<AudioSample> aplicarPerturbacion(const std::vector<AudioSample>& audio,
    TipoPerturbacion tipo,
    const ConfigAugmentation& config,
    unsigned int seed) {
    if (audio.empty()) {
        std::cerr << "! Error: Audio vacio en aplicarPerturbacion" << std::endl;
        return {};
    }

    // Obtener generador para numeros aleatorios
    auto& gen = obtenerGenerador(seed);
    std::uniform_real_distribution<AudioSample> distUniforme(0.0, 1.0);

    switch (tipo) {
    case PERTURBACION_ORIGINAL:
        // Sin perturbacion, solo copiar
        if (config.verbose) {
            std::cout << "   -> Original (sin perturbacion)" << std::endl;
        }
        return audio;

    case PERTURBACION_RUIDO: {
        // Ruido blanco con intensidad de configuracion
        if (config.verbose) {
            std::cout << "   -> Variacion 1: Ruido blanco (std="
                << config.intensidadRuido << ")" << std::endl;
        }
        return aplicarRuidoBlanco(audio, config.intensidadRuido, seed);
    }

    case PERTURBACION_VOLUMEN: {
        // Volumen aleatorio en rango configurado
        AudioSample factor = config.volumenMin +
            (config.volumenMax - config.volumenMin) * distUniforme(gen);
        if (config.verbose) {
            std::cout << "   -> Variacion 2: Volumen x" << factor << std::endl;
        }
        return aplicarEscaladoVolumen(audio, factor);
    }

    case PERTURBACION_RUIDO_VOLUMEN: {
        // Ruido + volumen combinado
        AudioSample intensidad = config.intensidadRuido * 0.7; // Ruido mas suave
        AudioSample volumen = config.volumenMin +
            (config.volumenMax - config.volumenMin) * distUniforme(gen);

        if (config.verbose) {
            std::cout << "   -> Variacion 3: Ruido+Volumen (std="
                << intensidad << ", vol=" << volumen << ")" << std::endl;
        }

        // Aplicar ruido primero
        auto conRuido = aplicarRuidoBlanco(audio, intensidad, seed);
        // Luego escalar volumen
        return aplicarEscaladoVolumen(conRuido, volumen);
    }

    case PERTURBACION_VELOCIDAD: {
        // Velocidad aleatoria en rango configurado
        AudioSample factor = config.velocidadMin +
            (config.velocidadMax - config.velocidadMin) * distUniforme(gen);
        if (config.verbose) {
            std::cout << "   -> Variacion 4: Velocidad x" << factor << std::endl;
        }
        return aplicarCambioVelocidad(audio, factor);
    }

    default:
        std::cerr << "! Error: Tipo de perturbacion desconocido: " << tipo << std::endl;
        return {};
    }
}

// ================================================================================
// FUNCION PRINCIPAL: GENERAR VARIACIONES
// ================================================================================

std::vector<std::vector<AudioSample>> generarVariacionesAudio(
    const std::vector<AudioSample>& audioOriginal,
    int numVariacionesExtra,
    ConfigAugmentation* config) {

    if (audioOriginal.empty()) {
        std::cerr << "! Error: Audio original vacio" << std::endl;
        return {};
    }

    if (numVariacionesExtra < 0 || numVariacionesExtra > 10) {
        std::cerr << "! Error: Numero de variaciones debe estar entre 0 y 10" << std::endl;
        return {};
    }

    // Usar configuracion global si no se proporciona
    const ConfigAugmentation& configActual = config ? *config : CONFIG_AUG;

    // Inicializar generador
    obtenerGenerador(configActual.seed);

    if (configActual.verbose) {
        std::cout << "\n-> Generando variaciones de audio" << std::endl;
        std::cout << "   Muestras originales: " << audioOriginal.size() << std::endl;
        std::cout << "   Numero de variaciones: " << numVariacionesExtra << std::endl;
    }

    // Vector de variaciones (original + variaciones)
    std::vector<std::vector<AudioSample>> variaciones;
    variaciones.reserve(numVariacionesExtra + 1);

    // Generar variaciones
    int totalVariaciones = numVariacionesExtra + 1;
    for (int i = 0; i < totalVariaciones; ++i) {
        TipoPerturbacion tipo;

        if (i == 0) {
            // Primera variacion: original sin perturbar
            tipo = PERTURBACION_ORIGINAL;
        }
        else {
            // Variaciones 1-4: diferentes perturbaciones
            tipo = static_cast<TipoPerturbacion>(i % 5);
            if (tipo == PERTURBACION_ORIGINAL) {
                tipo = PERTURBACION_RUIDO; // Evitar otro original
            }
        }

        auto variacion = aplicarPerturbacion(audioOriginal, tipo, configActual,
            configActual.seed + i);

        if (variacion.empty()) {
            std::cerr << "! Error: Fallo al generar variacion " << i << std::endl;
            return {}; // Retornar vacio en caso de error
        }

        variaciones.push_back(std::move(variacion));
    }

    if (configActual.verbose) {
        std::cout << "   & Variaciones generadas exitosamente" << std::endl;
        for (size_t i = 0; i < variaciones.size(); ++i) {
            std::cout << "   [" << i << "] " << variaciones[i].size()
                << " muestras" << std::endl;
        }
    }

    return variaciones;
}

// ================================================================================
// FUNCIONES DE DEBUG
// ================================================================================

void imprimirEstadisticasAudio(const std::vector<AudioSample>& audio,
    const std::string& nombre) {
    if (audio.empty()) {
        std::cerr << "! Error: Audio vacio" << std::endl;
        return;
    }

    // Calcular estadisticas
    AudioSample minVal = audio[0];
    AudioSample maxVal = audio[0];
    AudioSample suma = 0.0;

    for (const auto& sample : audio) {
        if (sample < minVal) minVal = sample;
        if (sample > maxVal) maxVal = sample;
        suma += sample;
    }

    AudioSample media = suma / audio.size();

    // Calcular desviacion estandar
    AudioSample sumaDesv = 0.0;
    for (const auto& sample : audio) {
        AudioSample diff = sample - media;
        sumaDesv += diff * diff;
    }
    AudioSample desviacion = std::sqrt(sumaDesv / audio.size());

    // Calcular RMS (Root Mean Square)
    AudioSample sumaRMS = 0.0;
    for (const auto& sample : audio) {
        sumaRMS += sample * sample;
    }
    AudioSample rms = std::sqrt(sumaRMS / audio.size());

    // Imprimir estadisticas
    std::cout << "\n-> Estadisticas de Audio: " << nombre << std::endl;
    std::cout << "   Muestras:      " << audio.size() << std::endl;
    std::cout << "   Rango:         [" << minVal << ", " << maxVal << "]" << std::endl;
    std::cout << "   Media:         " << media << std::endl;
    std::cout << "   Desv. Std:     " << desviacion << std::endl;
    std::cout << "   RMS:           " << rms << std::endl;

    // Detectar clipping
    int clippedPos = 0, clippedNeg = 0;
    for (const auto& sample : audio) {
        if (sample >= 0.99) clippedPos++;
        if (sample <= -0.99) clippedNeg++;
    }

    if (clippedPos > 0 || clippedNeg > 0) {
        std::cout << "   % Warning: Clipping detectado (+" << clippedPos
            << ", -" << clippedNeg << " muestras)" << std::endl;
    }
}
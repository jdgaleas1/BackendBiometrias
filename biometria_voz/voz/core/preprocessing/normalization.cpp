#include "preprocesar.h"
#include "../../utils/config.h"
#include "../../utils/additional.h"
#include <iomanip>  
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>


AudioSample calcularRMS(const std::vector<AudioSample>& audio) {
    if (audio.empty()) return 0.0;

    // Acumulacion directa en double sin conversiones
    double sumSquares = 0.0;

#if ENABLE_OPENMP
#pragma omp parallel for reduction(+:sumSquares)
#endif
    for (int i = 0; i < static_cast<int>(audio.size()); ++i) {
        sumSquares += audio[i] * audio[i];
    }

    return std::sqrt(sumSquares / audio.size());
}

AudioSample encontrarPico(const std::vector<AudioSample>& audio) {
    if (audio.empty()) return 0.0;

    AudioSample maxAbs = 0.0;

    // NOMINMAX impide usar reduction(max) en OpenMP, calcular max secuencialmente
    for (int i = 0; i < static_cast<int>(audio.size()); ++i) {
        AudioSample absVal = std::abs(audio[i]);
        if (absVal > maxAbs) maxAbs = absVal;
    }

    return maxAbs;
}

// ============================================================================
// NORMALIZACION RMS 

std::vector<AudioSample> normalizeRMS(
    const std::vector<AudioSample>& audio,
    AudioSample targetRMS)
{
    if (audio.empty()) {
        std::cerr << "! Error: Audio invalido en normalizeRMS" << std::endl;
        return {};
    }

    if (targetRMS <= 0.0 || targetRMS > 1.0) {
        std::cerr << "! Warning: targetRMS invalido, usando 0.1" << std::endl;
        targetRMS = 0.1;
    }

    // Calcular RMS actual
    AudioSample currentRMS = calcularRMS(audio);

    if (CONFIG_PREP.verbose) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "[ETAPA 2/6] PREPROCESAMIENTO - NORMALIZACION RMS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }

    // Umbral mas estricto con double
    if (currentRMS < 1e-12) {
        std::cout << "   % Warning: Audio practicamente silencioso (RMS < 1e-12)" << std::endl;
        return audio;  // Retornar copia sin modificar
    }

    // Calcular factor de ganancia
    AudioSample gain = targetRMS / currentRMS;

    if (CONFIG_PREP.verbose) {
        std::cout << "\n# Parametros de normalizacion" << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Metrica                | Umbral/Esperado        | Resultado        " << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | RMS actual             | -                      | " << std::setw(16) << currentRMS << std::endl;
        std::cout << "   | RMS objetivo           | " << std::setw(22) << targetRMS << " | (configurado)    " << std::endl;
        std::cout << "   | Factor de ganancia     | [0.1, 10.0]            | " << std::setw(15) << gain << "x " << std::endl;
    }

    // Aplicar ganancia con soft clipping
    std::vector<AudioSample> output;
    output.reserve(audio.size());

    for (size_t i = 0; i < audio.size(); ++i) {
        AudioSample normalized = audio[i] * gain;

        // Soft clipping si excede ±1.0 (preservar forma de onda)
        if (normalized > 1.0) {
            normalized = 1.0 - 0.1 * (1.0 - std::tanh((normalized - 1.0) * 2.0));
        }
        else if (normalized < -1.0) {
            normalized = -1.0 + 0.1 * (1.0 - std::tanh((-normalized - 1.0) * 2.0));
        }

        output.push_back(normalized);
    }

    // Verificar RMS final
    if (CONFIG_PREP.verbose) {
        AudioSample finalRMS = calcularRMS(output);
        AudioSample finalPeak = encontrarPico(output);
        std::cout << "\n# Resultado de normalizacion" << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Metrica                | Umbral/Esperado        | Resultado        " << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | RMS final              | ~" << std::setw(21) << targetRMS << " | " << std::setw(16) << finalRMS << std::endl;
        std::cout << "   | Pico maximo            | <= 1.0                 | " << std::setw(16) << finalPeak << std::endl;
    }

    return output;
}

// ============================================================================
// NORMALIZACION POR PICO 
// ============================================================================

std::vector<AudioSample> normalizePeak(
    const std::vector<AudioSample>& audio,
    AudioSample targetPeak)
{
    if (audio.empty()) {
        std::cerr << "! Error: Audio invalido en normalizePeak" << std::endl;
        return {};
    }

    if (targetPeak <= 0.0 || targetPeak > 1.0) {
        std::cerr << "! Warning: targetPeak invalido, usando 0.95" << std::endl;
        targetPeak = 0.95;
    }

    if (CONFIG_PREP.verbose) {
        std::cout << "-> Normalizando Peak (target=" << targetPeak << ")" << std::endl;
    }

    AudioSample currentPeak = encontrarPico(audio);

    if (currentPeak < 1e-12) {
        std::cout << "   % Warning: Audio practicamente silencioso" << std::endl;
        return audio;
    }

    AudioSample gain = targetPeak / currentPeak;

    if (CONFIG_PREP.verbose) {
        std::cout << "   Peak actual: " << currentPeak
            << " | Ganancia: " << gain << "x" << std::endl;
    }

    // Aplicar ganancia
    std::vector<AudioSample> output;
    output.reserve(audio.size());

    for (size_t i = 0; i < audio.size(); ++i) {
        output.push_back(audio[i] * gain);
    }

    if (CONFIG_PREP.verbose) {
        std::cout << "   & Peak normalization completada" << std::endl;
    }

    return output;
}

// ============================================================================
// CONTROL AUTOMATICO DE GANANCIA (AGC) 
// ============================================================================

std::vector<AudioSample> automaticGainControl(
    const std::vector<AudioSample>& audio,
    int sampleRate,
    int windowMs,
    AudioSample targetRMS)
{
    if (audio.empty()) {
        std::cerr << "! Error: Audio invalido en AGC" << std::endl;
        return {};
    }

    if (CONFIG_PREP.verbose) {
        std::cout << "-> Aplicando AGC (ventana=" << windowMs << "ms, targetRMS="
            << targetRMS << ")" << std::endl;
    }

    const int windowSamples = (windowMs * sampleRate) / 1000;
    const int hopSize = windowSamples / 2;  // 50% overlap

    if (windowSamples <= 0 || windowSamples > static_cast<int>(audio.size())) {
        std::cout << "   % Warning: Ventana invalida, usando RMS global" << std::endl;
        return normalizeRMS(audio, targetRMS);
    }

    const size_t numSamples = audio.size();

    // Inicializar buffers
    std::vector<AudioSample> output(numSamples, 0.0);
    std::vector<AudioSample> weightSum(numSamples, 0.0);

    const int numWindows = 1 + (static_cast<int>(numSamples) - windowSamples) / hopSize;

    if (CONFIG_PREP.verbose) {
        std::cout << "   Procesando " << numWindows << " ventanas" << std::endl;
    }

    // Procesar cada ventana
    for (int w = 0; w < numWindows; ++w) {
        int start = w * hopSize;
        int len = std::min(windowSamples, static_cast<int>(numSamples) - start);

        // Calcular RMS de esta ventana
        double sumSquares = 0.0;
        for (int i = 0; i < len; ++i) {
            AudioSample sample = audio[start + i];
            sumSquares += sample * sample;
        }
        AudioSample windowRMS = std::sqrt(sumSquares / len);

        // Calcular ganancia para esta ventana
        AudioSample gain = 1.0;
        if (windowRMS > 1e-4) {  // Umbral minimo para evitar amplificar ruido
            gain = targetRMS / windowRMS;
            gain = std::min(gain, 10.0);  // Limitar ganancia maxima
        }

        // Aplicar ganancia con ventana de Hanning para suavizar transiciones
        for (int i = 0; i < len; ++i) {
            AudioSample hanningWeight = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (len - 1)));
            output[start + i] += audio[start + i] * gain * hanningWeight;
            weightSum[start + i] += hanningWeight;
        }
    }

    // Normalizar por suma de pesos (overlap-add)
    for (size_t i = 0; i < numSamples; ++i) {
        if (weightSum[i] > 1e-6) {
            output[i] /= weightSum[i];
        }
        else {
            output[i] = audio[i];  // Fallback
        }

        // Clipping suave
        if (output[i] > 1.0) output[i] = 1.0;
        if (output[i] < -1.0) output[i] = -1.0;
    }

    if (CONFIG_PREP.verbose) {
        AudioSample finalRMS = calcularRMS(output);
        std::cout << "   & AGC completado | RMS final: " << finalRMS << std::endl;
    }

    return output;
}
#include "preprocesar.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>  

// VAD AVANZADO 
// Analisis multicaracteristica (energia, ZCR, entropia) sin denoising previo
std::vector<AudioSample> applyVAD(const std::vector<AudioSample>& audio,
    int sampleRate) {
    if (audio.empty()) {
        std::cerr << "! Error: Audio invalido en VAD avanzado" << std::endl;
        return {};
    }

    const auto& cfg = CONFIG_PREP;
    const int totalSamples = static_cast<int>(audio.size());

    const int frameSize = std::max(1, cfg.vadFrameSizeMs * sampleRate / 1000);
    const int stride = std::max(1, cfg.vadFrameStrideMs * sampleRate / 1000);
    const int paddingSamples = std::max(0, cfg.vadPaddingMs * sampleRate / 1000);
    const int minDurationSamples = std::max(1, cfg.vadMinDurationMs * sampleRate / 1000);
    const int mergeGapSamples = std::max(0, cfg.vadMergeGapMs * sampleRate / 1000);
    const int samplesPerBin = std::max(1, frameSize / 8);
    constexpr int entropyBins = 8;
    constexpr double kEps = 1e-12;

    // Validacion: audio muy corto se retorna completo
    if (totalSamples < frameSize) {
        if (cfg.verbose) {
            std::cout << "   % VAD: Audio muy corto (" << totalSamples
                << " samples), retornando completo" << std::endl;
        }
        return audio;
    }

    const int numFrames = 1 + (totalSamples - frameSize) / stride;

    struct FrameFeatures {
        double energy;
        double zcr;
        double entropy;
    };

    std::vector<FrameFeatures> frames;
    frames.reserve(numFrames);

    std::vector<double> energyValues;
    energyValues.reserve(numFrames);
    double sumEnergy = 0.0;
    double sumZcr = 0.0;
    double sumEntropy = 0.0;

    for (int frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
        const int start = frameIdx * stride;
        std::array<double, entropyBins> binEnergy{};
        double energySum = 0.0;
        double zeroCrossings = 0.0;
        double prevSample = audio[start];

        for (int j = 0; j < frameSize && start + j < totalSamples; ++j) {
            const double sample = audio[start + j];
            energySum += sample * sample;

            if (j > 0 && ((sample >= 0 && prevSample < 0) || (sample < 0 && prevSample >= 0))) {
                zeroCrossings += 1.0;
            }
            prevSample = sample;

            const int binIdx = std::min(entropyBins - 1, j / samplesPerBin);
            binEnergy[binIdx] += sample * sample;
        }

        const double frameEnergy = std::sqrt(energySum / std::max(1, frameSize));
        const double frameZcr = (frameSize > 1) ? zeroCrossings / (frameSize - 1) : 0.0;

        double totalBinEnergy = 0.0;
        for (double value : binEnergy) {
            totalBinEnergy += value;
        }
        totalBinEnergy = std::max(totalBinEnergy, kEps);

        double spectralEntropy = 0.0;
        for (double value : binEnergy) {
            if (value <= 0.0) {
                continue;
            }
            const double p = value / totalBinEnergy;
            spectralEntropy -= p * std::log2(p);
        }
        spectralEntropy /= std::log2(static_cast<double>(entropyBins));

        frames.push_back({ frameEnergy, frameZcr, spectralEntropy });
        energyValues.push_back(frameEnergy);
        sumEnergy += frameEnergy;
        sumZcr += frameZcr;
        sumEntropy += spectralEntropy;
    }

    std::vector<double> sortedEnergy = energyValues;
    std::sort(sortedEnergy.begin(), sortedEnergy.end());
    const double medianEnergy = (sortedEnergy.size() % 2 == 0 && !sortedEnergy.empty())
        ? 0.5 * (sortedEnergy[sortedEnergy.size() / 2] +
            sortedEnergy[sortedEnergy.size() / 2 - 1])
        : (sortedEnergy.empty() ? 0.0 : sortedEnergy[sortedEnergy.size() / 2]);

    const double meanEnergy = (numFrames > 0) ? sumEnergy / numFrames : 0.0;
    const double meanZcr = (numFrames > 0) ? sumZcr / numFrames : 0.0;
    const double meanEntropy = (numFrames > 0) ? sumEntropy / numFrames : 0.0;

    const double energyThreshold = std::max(cfg.vadEnergyThreshold,
        std::max(medianEnergy * 0.75, meanEnergy * 0.6));
    const double zcrThreshold = std::max(0.02, meanZcr * 0.9);
    const double entropyThreshold = std::max(0.05, meanEntropy * 0.95);

    if (cfg.verbose) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "[ETAPA 2/6] PREPROCESAMIENTO - VAD (Voice Activity Detection)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\n# Configuracion de frames" << std::endl;
        std::cout << "   Total frames: " << numFrames << std::endl;
        std::cout << "   Tamanho frame: " << frameSize << " samples (" << cfg.vadFrameSizeMs << "ms)" << std::endl;
        std::cout << "   Stride: " << stride << " samples (" << cfg.vadFrameStrideMs << "ms)" << std::endl;
        std::cout << "\n# Umbrales adaptativos calculados" << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Caracteristica         | Resultado calculado       | Base             " << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Energia (RMS)          | " << std::setw(22) << energyThreshold << " | Media/Mediana    " << std::endl;
        std::cout << "   | ZCR (cruces por cero)  | " << std::setw(22) << zcrThreshold << " | Media adaptada   " << std::endl;
        std::cout << "   | Entropia espectral     | " << std::setw(22) << entropyThreshold << " | Media adaptada   " << std::endl;
    }

    std::vector<bool> isVoice(numFrames, false);
    int framesRuidoPuro = 0;
    int framesVoz = 0;

    for (int i = 0; i < numFrames; ++i) {
        const auto& feat = frames[i];
        const bool energyGate = feat.energy >= energyThreshold;
        const bool relaxedEnergy = feat.energy >= energyThreshold * 0.5;
        const bool zcrGate = feat.zcr <= zcrThreshold * 1.15;
        const bool entropyGate = feat.entropy <= entropyThreshold * 1.1;

        bool voice = (energyGate && (zcrGate || entropyGate));
        if (!voice && relaxedEnergy) {
            voice = (feat.zcr <= zcrThreshold * 0.9 &&
                feat.entropy <= entropyThreshold);
        }

        isVoice[i] = voice;
        if (voice) {
            framesVoz++;
        }
        else {
            framesRuidoPuro++;
        }
    }

    const double percentVoice = (numFrames > 0) ? 100.0 * framesVoz / numFrames : 0.0;

    if (cfg.verbose) {
        std::cout << "\n# Resultado de deteccion" << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Metrica                | Umbral/Esperado        | Resultado        " << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Frames con voz         | > 20%                  | " << std::setw(11) << framesVoz << "/" << std::setw(4) << numFrames << std::endl;
        std::cout << "   | Porcentaje voz (%)     | > 20%                  | " << std::setw(15) << percentVoice << "% " << std::endl;
        std::cout << "   | Ruido descartado       | -                      | " << std::setw(10) << framesRuidoPuro << " frames " << std::endl;
    }

    // PASO 2: SUAVIZADO TEMPORAL
    std::vector<bool> smoothed = isVoice;

    // Llenar gaps de 1 frame
    for (int i = 1; i < numFrames - 1; ++i) {
        if (!isVoice[i] && isVoice[i - 1] && isVoice[i + 1]) {
            smoothed[i] = true;
        }
    }

    // Llenar gaps de 2 frames
    for (int i = 2; i < numFrames - 2; ++i) {
        if (!smoothed[i] && !smoothed[i - 1] && smoothed[i - 2] && smoothed[i + 1]) {
            smoothed[i] = true;
            smoothed[i - 1] = true;
        }
    }

    isVoice = smoothed;

    // PASO 3: EXTRAER SEGMENTOS
    std::vector<std::pair<int, int>> segments;
    int i = 0;

    while (i < numFrames) {
        if (!isVoice[i]) {
            ++i;
            continue;
        }

        int startFrame = i;
        while (i < numFrames && isVoice[i]) ++i;
        int endFrame = i;

        int startSample = startFrame * stride - paddingSamples;
        int endSample = endFrame * stride + frameSize + paddingSamples;
        startSample = std::max(0, startSample);
        endSample = std::min(totalSamples, endSample);

        if (endSample - startSample >= minDurationSamples) {
            segments.emplace_back(startSample, endSample);
        }
    }

    if (segments.empty()) {
        if (cfg.verbose) {
            std::cout << "   % Warning: No se detecto voz, retornando audio completo"
                << std::endl;
        }
        return audio;
    }


    // PASO 4: MERGE SEGMENTOS CERCANOS
    std::vector<std::pair<int, int>> merged;
    merged.reserve(segments.size());
    merged.push_back(segments[0]);

    for (size_t s = 1; s < segments.size(); ++s) {
        int gap = segments[s].first - merged.back().second;
        if (gap <= mergeGapSamples) {
            merged.back().second = std::max(merged.back().second, segments[s].second);
        }
        else {
            merged.push_back(segments[s]);
        }
    }

    // PASO 5: CONCATENAR RESULTADO
    int totalKeep = 0;
    for (const auto& seg : merged) {
        totalKeep += seg.second - seg.first;
    }

    if (totalKeep <= 0) {
        return {};
    }

    std::vector<AudioSample> result;
    result.reserve(totalKeep);

    for (const auto& seg : merged) {
        result.insert(result.end(),
            audio.begin() + seg.first,
            audio.begin() + seg.second);
    }

    double retention = 100.0 * totalKeep / totalSamples;

    if (cfg.verbose) {
        std::cout << "   & VAD completado | Retencion: " << retention
            << "% (" << result.size() << "/" << totalSamples << " samples)"
            << std::endl;
    }

    return result;
}
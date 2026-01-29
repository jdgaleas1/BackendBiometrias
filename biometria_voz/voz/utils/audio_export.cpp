#include "audio_export.h"
#include "external/dr_wav.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// ============================================================================
// EXPORTACION DE AUDIO WAV
// ============================================================================

void saveAudioToWav(const char* wavFile, const AudioSample* audioData, 
                    int numSamples, int sampleRate, int numChannels) {
    
    // Configurar formato WAV (32-bit float es requerido por dr_wav)
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;  // 32-bit float
    format.channels = numChannels;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 32;

    drwav wav;
    if (!drwav_init_file_write(&wav, wavFile, &format, nullptr)) {
        std::cerr << "! Error al abrir archivo WAV: " << wavFile << std::endl;
        return;
    }

    // Conversion AudioSample (double) → float SOLO al escribir
    // Mantenemos precision double en memoria hasta el ultimo momento
    std::vector<float> tempBuffer(numSamples);
    
    for (int i = 0; i < numSamples; ++i) {
        AudioSample val = audioData[i];
        
        // Verificar valores validos
        if (std::isnan(val) || std::isinf(val)) {
            tempBuffer[i] = 0.0f;  // Reemplazar invalidos con silencio
        } else {
            // Clipping sin distorsion
            tempBuffer[i] = static_cast<float>(std::max(-1.0, std::min(1.0, val)));
        }
    }

    // Escribir frames
    drwav_write_pcm_frames(&wav, numSamples / numChannels, tempBuffer.data());
    drwav_uninit(&wav);

    std::cout << "[WAV] Guardado (32-bit float, precision: double → float al escribir): " 
              << wavFile << std::endl;
}

// ============================================================================
// EXPORTACION DE ESPECTROGRAMA CSV
// ============================================================================

bool exportarEspectrogramaCSV(const std::vector<std::vector<AudioSample>>& spectrogram,
                               int sampleRate, 
                               int fftSize,
                               const char* filepath) {
    
    if (spectrogram.empty()) {
        std::cerr << "! Error: Espectrograma vacio" << std::endl;
        return false;
    }

    int frames = static_cast<int>(spectrogram.size());
    int bins = static_cast<int>(spectrogram[0].size());

    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << filepath << std::endl;
        return false;
    }

    std::cout << "-> Exportando espectrograma a CSV: " << filepath << std::endl;
    std::cout << "   Dimensiones: " << frames << " frames x " << bins << " bins" << std::endl;
    std::cout << "   Precision: AudioSample (double) → float al escribir CSV" << std::endl;

    // Header: frecuencias en Hz para cada bin
    out << "frame";
    for (int b = 0; b < bins; ++b) {
        double freq = static_cast<double>(b) * sampleRate / fftSize;
        out << ",freq_" << std::fixed << std::setprecision(1) << freq << "_Hz";
    }
    out << "\n";

    // Datos: cada fila es un frame temporal
    // Conversion AudioSample → float ocurre AQUI (al escribir)
    for (int f = 0; f < frames; ++f) {
        out << f;
        for (int b = 0; b < bins; ++b) {
            // Conversion a float SOLO al escribir (precision double en memoria)
            float value = static_cast<float>(spectrogram[f][b]);
            out << "," << std::scientific << std::setprecision(6) << value;
        }
        out << "\n";
    }

    out.close();
    std::cout << "   & CSV exportado: " << frames << " frames x " << bins << " bins" << std::endl;
    return true;
}

// ============================================================================
// EXPORTACION DE MFCC CSV
// ============================================================================

bool exportarMFCC_CSV(const std::vector<std::vector<AudioSample>>& mfcc,
                      const char* filepath) {
    
    if (mfcc.empty()) {
        std::cerr << "! Error: MFCC vacio" << std::endl;
        return false;
    }

    int frames = static_cast<int>(mfcc.size());
    int coeffs = static_cast<int>(mfcc[0].size());

    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << filepath << std::endl;
        return false;
    }

    std::cout << "-> Exportando MFCC a CSV: " << filepath << std::endl;
    std::cout << "   Dimensiones: " << frames << " frames x " << coeffs << " coeficientes" << std::endl;
    std::cout << "   Precision: AudioSample (double) → float al escribir CSV" << std::endl;

    // Header: coeficientes MFCC
    out << "frame";
    for (int c = 0; c < coeffs; ++c) {
        out << ",mfcc_" << c;
    }
    out << "\n";

    // Datos: cada fila es un frame temporal
    // Conversion AudioSample → float ocurre AQUI (al escribir)
    for (int f = 0; f < frames; ++f) {
        out << f;
        for (int c = 0; c < coeffs; ++c) {
            // Conversion a float SOLO al escribir
            float value = static_cast<float>(mfcc[f][c]);
            out << "," << std::fixed << std::setprecision(6) << value;
        }
        out << "\n";
    }

    out.close();
    std::cout << "   & CSV exportado: " << frames << " frames x " << coeffs << " coeficientes" << std::endl;
    return true;
}

// ============================================================================
// EXPORTACION DE ESTADISTICAS MFCC CSV
// ============================================================================

bool exportarEstadisticasMFCC_CSV(const std::vector<AudioSample>& stats, 
                                   const char* filepath) {
    
    if (stats.empty()) {
        std::cerr << "! Error: Estadisticas vacias" << std::endl;
        return false;
    }

    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << filepath << std::endl;
        return false;
    }

    std::cout << "-> Exportando estadisticas MFCC a CSV: " << filepath << std::endl;
    std::cout << "   Features: " << stats.size() << std::endl;
    std::cout << "   Precision: AudioSample (double) → float al escribir CSV" << std::endl;

    // Header
    out << "feature_index,value,tipo\n";

    // Datos: vector de estadisticas (MEAN de cada coeficiente)
    // Conversion AudioSample → float ocurre AQUI
    for (size_t i = 0; i < stats.size(); ++i) {
        float value = static_cast<float>(stats[i]);
        out << i << "," << std::fixed << std::setprecision(6) << value << ",mean\n";
    }

    out.close();
    std::cout << "   & CSV exportado: " << stats.size() << " features" << std::endl;
    return true;
}
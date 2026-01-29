#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include "../../../core/load_audio/audio_io.h"
#include "../../../core/preprocessing/preprocesar.h"
#include "../../../utils/audio_export.h"
#include "../../../utils/config.h"

namespace fs = std::filesystem;

void exportarAudios() {
    std::string carpetaOrigen = "D:\\testDataset";
    std::string carpetaSalida = "exportar_audios";

    std::cout << "-> EXPORTADOR DE AUDIOS - PIPELINE ROBUSTO COMPLETO <-" << std::endl;
    std::cout << "\nCarpeta origen: " << carpetaOrigen << std::endl;
    std::cout << "Carpeta salida: " << carpetaSalida << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Pipeline aplicado (ORDEN OPTIMIZADO):" << std::endl;
    std::cout << "  1. Original" << std::endl;
    std::cout << "  2. Normalizacion (RMS=" << CONFIG_PREP.normalizationTargetRMS << ") - Estabiliza amplitudes" << std::endl;
    std::cout << "  3. VAD Avanzado standalone (Energia+ZCR+Entropia)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // Mostrar configuracion global que se esta usando (igual que audio_pipeline.cpp)
    std::cout << "\n@ Configuracion de preprocesamiento (CONFIG_PREP):" << std::endl;
    CONFIG_PREP.mostrar();
    std::cout << std::string(50, '-') << std::endl;

    // Crear carpeta de salida si no existe
    if (!fs::exists(carpetaSalida)) {
        std::cout << "@ Creando carpeta de salida: " << carpetaSalida << std::endl;
        fs::create_directories(carpetaSalida);
    }

    // Verificar que la carpeta existe
    if (!fs::exists(carpetaOrigen)) {
        std::cout << "% ERROR: La carpeta no existe: " << carpetaOrigen << std::endl;
        std::cout << "   Verifica la ruta" << std::endl;
        std::cout << "\nPresiona cualquier tecla para cerrar..." << std::endl;
        std::cin.get();
        return;
    }

    if (!fs::is_directory(carpetaOrigen)) {
        std::cout << "% ERROR: La ruta no es una carpeta: " << carpetaOrigen << std::endl;
        std::cout << "\nPresiona cualquier tecla para cerrar..." << std::endl;
        std::cin.get();
        return;
    }

    // Extensiones de audio soportadas
    std::vector<std::string> extensionesValidas = {".wav", ".mp3", ".flac", ".ogg"};

    // Recopilar archivos de audio
    std::vector<fs::path> archivosAudio;
    for (const auto& entry : fs::directory_iterator(carpetaOrigen)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (std::find(extensionesValidas.begin(), extensionesValidas.end(), ext) != extensionesValidas.end()) {
                archivosAudio.push_back(entry.path());
            }
        }
    }

    if (archivosAudio.empty()) {
        std::cout << "% No se encontraron archivos de audio en la carpeta" << std::endl;
        std::cout << "   Extensiones soportadas: .wav, .mp3, .flac, .ogg" << std::endl;
        std::cout << "\nPresiona cualquier tecla para cerrar..." << std::endl;
        std::cin.get();
        return;
    }

    std::cout << "@ Encontrados " << archivosAudio.size() << " archivos de audio" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // Procesar cada archivo
    int exitosos = 0;
    int fallidos = 0;

    for (size_t i = 0; i < archivosAudio.size(); ++i) {
        const auto& archivoPath = archivosAudio[i];
        std::string nombreArchivo = archivoPath.filename().string();
        
        std::cout << "\n[" << (i + 1) << "/" << archivosAudio.size() << "] " << nombreArchivo << std::endl;

        // Cargar audio usando loadAudio
        int sr, numChannels, numSamples;
        std::vector<AudioSample> audio = loadAudio(archivoPath.string().c_str(), sr, numChannels, numSamples);
        
        if (audio.empty()) {
            std::cout << "  % ERROR: No se pudo cargar el archivo" << std::endl;
            fallidos++;
            continue;
        }

        if (numSamples == 0) {
            std::cout << "  % ERROR: Audio vacio" << std::endl;
            fallidos++;
            continue;
        }

        std::cout << "  @ Cargado: " << numSamples << " muestras, " << sr << " Hz" << std::endl;
        std::cout << "  Duracion original: " << (numSamples / static_cast<double>(sr)) << " segundos" << std::endl;

        // Prefijo base para archivos
        std::string prefijoBase = carpetaSalida + "\\" + archivoPath.stem().string();

        std::cout << "\n  -> PIPELINE ROBUSTO DE PREPROCESAMIENTO <-" << std::endl;

        // === PASO 1: EXPORTAR ORIGINAL ===
        std::cout << "\n  [1/3] ORIGINAL" << std::endl;
        std::string nombreOriginal = prefijoBase + "_1_original.wav";
        saveAudioToWav(nombreOriginal.c_str(), audio.data(), numSamples, sr, 1);
        std::cout << "        Exportado: " << nombreOriginal << std::endl;

        // === PASO 2: NORMALIZACION RMS ===
        std::cout << "\n  [2/3] NORMALIZACION (RMS=" << CONFIG_PREP.normalizationTargetRMS << ")" << std::endl;
        std::vector<AudioSample> normalized = normalizeRMS(audio, CONFIG_PREP.normalizationTargetRMS);
        if (normalized.empty()) {
            std::cout << "         Error en normalizacion" << std::endl;
            fallidos++;
            continue;
        }
        std::string nombreNormalized = prefijoBase + "_2_normalized.wav";
        saveAudioToWav(nombreNormalized.c_str(), normalized.data(), static_cast<int>(normalized.size()), sr, 1);
        std::cout << "         Exportado: " << nombreNormalized << std::endl;
        std::cout << "        Efecto: Volumen constante RMS=" << CONFIG_PREP.normalizationTargetRMS 
                  << ", optimiza etapas posteriores" << std::endl;

        // === PASO 3: VAD AVANZADO ===
        std::cout << "\n  [3/3] VAD AVANZADO (Energia+ZCR+Entropia)" << std::endl;
        std::vector<AudioSample> voz = applyVAD(normalized, sr);
        
        if (voz.empty()) {
            std::cout << "        No se detecto voz activa (audio descartado)" << std::endl;
            fallidos++;
            continue;
        }
        std::string nombreVAD = prefijoBase + "_3_vad_final.wav";
        // Precision completa hasta escritura WAV
        saveAudioToWav(nombreVAD.c_str(), voz.data(), static_cast<int>(voz.size()), sr, 1);
        
        std::cout << "        Exportado: " << nombreVAD << std::endl;
        std::cout << "        Efecto: Solo voz activa (Energia+ZCR+Entropia), ~90% precision" << std::endl;
        std::cout << "        Retencion total: " << (100.0 * voz.size() / numSamples) << "%" 
                  << " (del original)" << std::endl;
        std::cout << "        Duracion final: " << (voz.size() / static_cast<double>(sr)) << " segundos" << std::endl;
        std::cout << "        Detecta: consonantes no sonoras, rechaza ruido blanco/tonal" << std::endl;

        std::cout <<   " PIPELINE COMPLETADO - 3 ARCHIVOS EXPORTADOS   " << std::endl;
        std::cout << "        @ Precision AudioSample (double) hasta conversion WAV final" << std::endl;
        exitosos++;
    }

    // Resumen final
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "RESUMEN FINAL " << std::endl;

    std::cout << "\nArchivos procesados: " << archivosAudio.size() << std::endl;
    std::cout << "  Exitosos: " << exitosos << std::endl;
    std::cout << "  Fallidos: " << fallidos << std::endl;
    
    if (exitosos > 0) {
        std::cout << "\nUbicacion: " << fs::absolute(carpetaSalida) << std::endl;
        std::cout << "\n Archivos exportados por audio (ORDEN OPTIMIZADO):" << std::endl;
        std::cout << "  1. [nombre]_1_original.wav     - Audio original sin procesar" << std::endl;
        std::cout << "  2. [nombre]_2_normalized.wav   - Despues de normalizacion RMS (RMS=" 
              << CONFIG_PREP.normalizationTargetRMS << ")" << std::endl;
        std::cout << "  3. [nombre]_3_vad_final.wav    - Despues de VAD standalone (solo voz)" << std::endl;
        std::cout << "\n Total de archivos: " << (exitosos * 3) << " archivos WAV" << std::endl;
    }
    
    std::cout << std::string(60, '=') << std::endl;
}

int main() {
    auto inicio = std::chrono::high_resolution_clock::now();
    exportarAudios();

        // TIEMPO =====================================================================
    auto fin = std::chrono::high_resolution_clock::now();
    auto duracion = std::chrono::duration_cast<std::chrono::seconds>(fin - inicio);

    std::cout << "\n@ Tiempo de exportarr: " << duracion.count() << " segundos" << std::endl;

    return 0;
}
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include "../../../core/load_audio/audio_io.h"
#include "../../../core/preprocessing/preprocesar.h"
#include "../../../core/segmentation/stft.h"
#include "../../../core/features/mfcc.h"
#include "../../../utils/audio_export.h"
#include "../../../utils/config.h"

namespace fs = std::filesystem;

void exportarFeatures() {
    std::string carpetaOrigen = "D:\\testDataset";
    std::string carpetaSalida = "exportar_features";

    std::cout << std::string(70, '=') << std::endl;
    std::cout << "  EXPORTADOR DE FEATURES - STFT Y MFCC PARA ANALISIS PYTHON  " << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "\nCarpeta origen: " << carpetaOrigen << std::endl;
    std::cout << "Carpeta salida: " << carpetaSalida << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Pipeline de exportacion:" << std::endl;
    std::cout << "  1. Preprocesamiento (Normalizacion + VAD standalone)" << std::endl;
    std::cout << "  2. STFT (Espectrograma) -> CSV" << std::endl;
    std::cout << "  3. MFCC (Coeficientes por frame) -> CSV" << std::endl;
    std::cout << "  4. Estadisticas MFCC (Features finales) -> CSV" << std::endl;
    std::cout << "  @ Precision: AudioSample (double) hasta escritura de CSV" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Mostrar configuracion global
    std::cout << "\n@ Configuracion del sistema:" << std::endl;
    CONFIG_PREP.mostrar();
    std::cout << std::endl;
    CONFIG_STFT.mostrar();
    std::cout << std::endl;
    CONFIG_MFCC.mostrar();
    std::cout << std::string(70, '-') << std::endl;

    // Crear carpeta de salida
    if (!fs::exists(carpetaSalida)) {
        std::cout << "@ Creando carpeta de salida: " << carpetaSalida << std::endl;
        fs::create_directories(carpetaSalida);
    }

    // Verificar carpeta origen
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

    // Extensiones soportadas
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
    std::cout << std::string(70, '-') << std::endl;

    // Procesar cada archivo
    int exitosos = 0;
    int fallidos = 0;

    for (size_t i = 0; i < archivosAudio.size(); ++i) {
        const auto& archivoPath = archivosAudio[i];
        std::string nombreArchivo = archivoPath.filename().string();
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "[" << (i + 1) << "/" << archivosAudio.size() << "] " << nombreArchivo << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // ========================================================================
        // PASO 1: CARGAR AUDIO
        // ========================================================================
        std::cout << "\n[PASO 1/4] CARGA DE AUDIO" << std::endl;
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

        std::cout << "  @ Cargado: " << numSamples << " muestras, " << sr << " Hz, "
                  << numChannels << " canal(es)" << std::endl;
        std::cout << "  @ Duracion: " << (numSamples / static_cast<double>(sr)) << " segundos" << std::endl;

        // ========================================================================
        // PASO 2: PREPROCESAMIENTO
        // ========================================================================
        std::cout << "\n[PASO 2/4] PREPROCESAMIENTO" << std::endl;
        
        // Normalizacion
        std::cout << "  -> Normalizacion RMS (target=" << CONFIG_PREP.normalizationTargetRMS << ")" << std::endl;
        std::vector<AudioSample> normalized = normalizeRMS(audio, CONFIG_PREP.normalizationTargetRMS);
        if (normalized.empty()) {
            std::cout << "     % ERROR en normalizacion" << std::endl;
            fallidos++;
            continue;
        }

        // VAD standalone
        std::cout << "  -> VAD Avanzado standalone (Energia+ZCR+Entropia)" << std::endl;
        std::vector<AudioSample> voz = applyVAD(normalized, sr);
        
        if (voz.empty()) {
            std::cout << "     % ERROR: No se detecto voz activa" << std::endl;
            fallidos++;
            continue;
        }

        int vadSamples = static_cast<int>(voz.size());
        std::cout << "  @ Audio preprocesado: " << vadSamples << " muestras ("
              << (vadSamples / static_cast<double>(sr)) << " seg)" << std::endl;
        std::cout << "  @ Retencion: " << (100.0 * vadSamples / numSamples) << "%" << std::endl;

        // ========================================================================
        // PASO 3: STFT (ESPECTROGRAMA)
        // ========================================================================
        std::cout << "\n[PASO 3/4] EXTRACCION STFT (ESPECTROGRAMA)" << std::endl;
        
        std::vector<std::vector<AudioSample>> spectrogram = applySTFT(voz, sr);
        
        if (spectrogram.empty()) {
            std::cout << "     % ERROR en STFT" << std::endl;
            fallidos++;
            continue;
        }

        int numFrames = static_cast<int>(spectrogram.size());
        int numBins = spectrogram.empty() ? 0 : static_cast<int>(spectrogram[0].size());
        std::cout << "  @ Espectrograma generado: " << numFrames << " frames x " << numBins << " bins" << std::endl;
        
        // Calcular FFT size para metadatos
        int frameSize = sr * CONFIG_STFT.frameSizeMs / 1000;
        int fftSize = 1;
        while (fftSize < frameSize) fftSize <<= 1;
        
        std::cout << "  @ FFT size: " << fftSize << " | Resolucion freq: "
                  << (sr / static_cast<double>(fftSize)) << " Hz/bin" << std::endl;

        // ========================================================================
        // PASO 4: MFCC
        // ========================================================================
        std::cout << "\n[PASO 4/4] EXTRACCION MFCC" << std::endl;
        
        std::vector<std::vector<AudioSample>> mfcc = extractMFCC(spectrogram, sr);
        
        if (mfcc.empty()) {
            std::cout << "     % ERROR en MFCC" << std::endl;
            fallidos++;
            continue;
        }

        int outFrames = static_cast<int>(mfcc.size());
        int outCoeffs = mfcc.empty() ? 0 : static_cast<int>(mfcc[0].size());
        std::cout << "  @ MFCC generado: " << outFrames << " frames x " << outCoeffs << " coeficientes" << std::endl;

        // Calcular estadisticas
        std::vector<AudioSample> estadisticas = calcularEstadisticasMFCC(mfcc);
        std::cout << "  @ Estadisticas calculadas: " << estadisticas.size() << " features" << std::endl;

        // ========================================================================
        // EXPORTACION CSV - SIN CONVERSIONES INNECESARIAS
        // ========================================================================
        std::cout << "\n[EXPORTACION] Guardando archivos CSV..." << std::endl;
        
        std::string prefijoBase = carpetaSalida + "\\" + archivoPath.stem().string();

        // Exportar espectrograma - DIRECTAMENTE desde std::vector (sin conversiones)
        std::string csvEspectrograma = prefijoBase + "_espectrograma.csv";
        if (!exportarEspectrogramaCSV(spectrogram, sr, fftSize, csvEspectrograma.c_str())) {
            std::cout << "  % Warning: Fallo exportar espectrograma" << std::endl;
        }

        // Exportar MFCC frames - DIRECTAMENTE desde std::vector (sin conversiones)
        std::string csvMFCC = prefijoBase + "_mfcc_frames.csv";
        if (!exportarMFCC_CSV(mfcc, csvMFCC.c_str())) {
            std::cout << "  % Warning: Fallo exportar MFCC frames" << std::endl;
        }

        // Exportar estadisticas - DIRECTAMENTE desde std::vector (sin conversiones)
        std::string csvStats = prefijoBase + "_mfcc_stats.csv";
        if (!exportarEstadisticasMFCC_CSV(estadisticas, csvStats.c_str())) {
            std::cout << "  % Warning: Fallo exportar estadisticas" << std::endl;
        }

        std::cout << "\n  # EXPORTACION COMPLETADA - 3 ARCHIVOS CSV GENERADOS #" << std::endl;
        std::cout << "    * " << archivoPath.stem().string() << "_espectrograma.csv" << std::endl;
        std::cout << "    * " << archivoPath.stem().string() << "_mfcc_frames.csv" << std::endl;
        std::cout << "    * " << archivoPath.stem().string() << "_mfcc_stats.csv" << std::endl;
        std::cout << "    @ Sin conversiones intermedias, sin punteros raw" << std::endl;
        std::cout << "    @ Precision double hasta escritura CSV" << std::endl;
        
        exitosos++;
    }

    // ========================================================================
    // RESUMEN FINAL
    // ========================================================================
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  RESUMEN FINAL  " << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\nArchivos procesados: " << archivosAudio.size() << std::endl;
    std::cout << "  Exitosos: " << exitosos << std::endl;
    std::cout << "  Fallidos: " << fallidos << std::endl;
    
    if (exitosos > 0) {
        std::cout << "\nUbicacion: " << fs::absolute(carpetaSalida) << std::endl;
        std::cout << "\nArchivos exportados por audio:" << std::endl;
        std::cout << "  * [nombre]_espectrograma.csv  - Matriz STFT (frames x bins frecuenciales)" << std::endl;
        std::cout << "  * [nombre]_mfcc_frames.csv    - Coeficientes MFCC por frame temporal" << std::endl;
        std::cout << "  * [nombre]_mfcc_stats.csv     - Features finales (mean de coeficientes)" << std::endl;
        std::cout << "\nTotal archivos CSV: " << (exitosos * 3) << std::endl;
    }
    
    std::cout << std::string(70, '=') << std::endl;
}

int main() {
    exportarFeatures();

    std::cout << "\nPresiona cualquier tecla para cerrar..." << std::endl;
    std::cin.get();

    return 0;
}
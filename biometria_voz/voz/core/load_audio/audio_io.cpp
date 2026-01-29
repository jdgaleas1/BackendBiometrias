#include "audio_io.h"
#include "../../utils/config.h"
#include "../../external/minimp3.h"
#include "../../external/minimp3_ex.h"
#include "../../external/dr_wav.h"
#include "../../external/dr_flac.h"
#include <cstring>
#include <string>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// ============================================================================
// CONSTANTES DE CONFIGURACION
// ============================================================================

// Limites de seguridad para prevenir problemas de memoria
const int MAX_SAMPLE_RATE = 48000;      // 48 kHz maximo
const int MIN_SAMPLE_RATE = 8000;       // 8 kHz minimo
const int MAX_CHANNELS = 2;             // Estereo maximo
const int MAX_DURATION_SECONDS = 300;   // 5 minutos maximo
const int MIN_SAMPLES = 1000;           // Minimo de muestras validas

// ============================================================================
// FUNCIONES DE VALIDACION
// ============================================================================

/**
 * Valida la calidad del audio cargado
 * Detecta: audio muy silencioso, muy ruidoso, o excede duracion maxima
 */
static bool validarCalidadAudio(const std::vector<AudioSample>& samples, int sampleRate, 
                                int channels, const char* contexto = "audio") {
    if (samples.empty()) {
        std::cerr << "! Error: Vector de audio vacio" << std::endl;
        return false;
    }

    int numSamples = static_cast<int>(samples.size());
    
    // VALIDACION 1: Duracion minima de 3.5 segundos para 69+ clases
    // Con muchas clases, audios cortos generan features poco discriminativos
    double duracionSegundos = static_cast<double>(numSamples) / (sampleRate * channels);
    const double MIN_DURACION_SEG = 3.5;  // CRITICO: Aumentado para 69+ clases
    
    if (duracionSegundos < MIN_DURACION_SEG) {
        std::cerr << "! Error: Audio demasiado corto: " << duracionSegundos 
                  << " segundos (minimo " << MIN_DURACION_SEG << "s)" << std::endl;
        return false;
    }
    
    // VALIDACION 2: Calcular RMS para detectar audio muy silencioso
    double sumSquares = 0.0;
    int clippedSamples = 0;
    
    for (int i = 0; i < numSamples; ++i) {
        double val = samples[i];
        sumSquares += val * val;
        
        // Contar muestras con clipping (muy ruidoso)
        if (std::abs(val) >= 0.99) {
            clippedSamples++;
        }
    }
    
    double rms = std::sqrt(sumSquares / numSamples);
    
    // Umbral minimo de RMS (audio muy silencioso)
    const double MIN_RMS = 0.005;
    if (rms < MIN_RMS) {
        std::cerr << "! Error: Audio demasiado silencioso (RMS=" << rms 
                  << ", minimo=" << MIN_RMS << ")" << std::endl;
        return false;
    }
    
    // VALIDACION 3: Audio muy ruidoso (exceso de clipping)
    double clippingPercentage = (100.0 * clippedSamples) / numSamples;
    const double MAX_CLIPPING_PERCENT = 5.0;  // Maximo 5% de clipping
    
    if (clippingPercentage > MAX_CLIPPING_PERCENT) {
        std::cerr << "! Error: Audio muy ruidoso - exceso de clipping (" 
                  << clippingPercentage << "%, maximo " << MAX_CLIPPING_PERCENT << "%)" << std::endl;
        return false;
    }
    
    // Audio aprobado
    double clippingPct = (100.0 * clippedSamples) / numSamples;
    
    std::cout << "\n# Validacion de calidad" << std::endl;
    std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
    std::cout << "   | Criterio               | Umbral/Esperado        | Resultado        " << std::endl;
    std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
    std::cout << "   | Duracion (s)           | >= " << std::setw(18) << MIN_DURACION_SEG << "s | " << std::setw(16) << duracionSegundos  << std::endl;
    std::cout << "   | RMS minimo             | >= " << std::setw(18) << MIN_RMS << " | " << std::setw(16) << rms << std::endl;
    std::cout << "   | Clipping (%)           | < " << std::setw(19) << MAX_CLIPPING_PERCENT << "% | " << std::setw(15) << clippingPct << "% " << std::endl;
   
    return true;
}

/**
 * Valida que los parametros de audio sean razonables
 */
static bool validarParametrosAudio(int sampleRate, int channels, int totalSamples) {
    if (sampleRate < MIN_SAMPLE_RATE || sampleRate > MAX_SAMPLE_RATE) {
        std::cerr << "! Error: Sample rate fuera de rango ["
            << MIN_SAMPLE_RATE << ", " << MAX_SAMPLE_RATE << " Hz]: "
            << sampleRate << " Hz" << std::endl;
        return false;
    }

    if (channels < 1 || channels > MAX_CHANNELS) {
        std::cerr << "! Error: Numero de canales invalido [1-"
            << MAX_CHANNELS << "]: " << channels << std::endl;
        return false;
    }

    if (totalSamples < MIN_SAMPLES) {
        std::cerr << "! Error: Archivo demasiado corto: "
            << totalSamples << " muestras (minimo "
            << MIN_SAMPLES << ")" << std::endl;
        return false;
    }

    // Calcular duracion y verificar limite
    int durationSeconds = totalSamples / (sampleRate * channels);
    if (durationSeconds > MAX_DURATION_SECONDS) {
        std::cerr << "! Error: Archivo demasiado largo: "
            << durationSeconds << " segundos (maximo "
            << MAX_DURATION_SECONDS << ")" << std::endl;
        return false;
    }

    return true;
}

/**
 * Valida que un archivo existe y es accesible
 */
static bool validarArchivo(const char* filePath) {
    if (filePath == nullptr || strlen(filePath) == 0) {
        std::cerr << "! Error: Ruta de archivo vacia o nula" << std::endl;
        return false;
    }

    // Verificar longitud de ruta razonable
    if (strlen(filePath) > 4096) {
        std::cerr << "! Error: Ruta de archivo demasiado larga" << std::endl;
        return false;
    }

    return true;
}

// ============================================================================
// DECODIFICADORES POR FORMATO - VERSION CON VECTORES
// ============================================================================

/**
 * Decodifica archivo WAV a vector de AudioSample
 * Formatos WAV soportados: PCM 8/16/24/32 bits, float 32/64 bits
 */
static std::vector<AudioSample> decodeWavToVector(const char* filePath, int& sampleRate,
    int& channels, int& totalSamples) {
    std::cout << "-> Decodificando archivo WAV: " << filePath << std::endl;

    drwav wav;
    if (!drwav_init_file(&wav, filePath, nullptr)) {
        std::cerr << "! Error: No se pudo abrir archivo WAV" << std::endl;
        return {};
    }

    sampleRate = static_cast<int>(wav.sampleRate);
    channels = static_cast<int>(wav.channels);
    drwav_uint64 totalFrames = wav.totalPCMFrameCount;
    totalSamples = static_cast<int>(totalFrames * channels);

    std::cout << "   Sample rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "   Canales: " << channels << std::endl;
    std::cout << "   Frames: " << totalFrames << std::endl;

    // Validar parametros antes de asignar memoria
    if (!validarParametrosAudio(sampleRate, channels, totalSamples)) {
        drwav_uninit(&wav);
        return {};
    }

    // Leer primero a buffer temporal float
    std::vector<float> tempBuffer(totalSamples);
    drwav_uint64 samplesRead = drwav_read_pcm_frames_f32(&wav, totalFrames, tempBuffer.data());

    drwav_uninit(&wav);

    if (samplesRead != totalFrames) {
        std::cerr << "% Warning: Leidas " << samplesRead << " frames de "
            << totalFrames << " esperadas" << std::endl;
        totalSamples = static_cast<int>(samplesRead * channels);
        tempBuffer.resize(totalSamples);
    }

    // Convertir de float a AudioSample (double) con precision completa
    std::vector<AudioSample> samples(totalSamples);
    
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < totalSamples; ++i) {
        samples[i] = static_cast<AudioSample>(tempBuffer[i]);
    }

    std::cout << "   & WAV decodificado exitosamente" << std::endl;
    return samples;
}

/**
 * Decodifica archivo FLAC a vector de AudioSample
 * FLAC es un formato de compresion sin perdidas
 */
static std::vector<AudioSample> decodeFlacToVector(const char* filePath, int& sampleRate,
    int& channels, int& totalSamples) {
    
    drflac* pFlac = drflac_open_file(filePath, nullptr);
    if (!pFlac) {
        std::cerr << "! Error: No se pudo abrir archivo FLAC" << std::endl;
        return {};
    }

    sampleRate = static_cast<int>(pFlac->sampleRate);
    channels = static_cast<int>(pFlac->channels);
    drflac_uint64 totalFrames = pFlac->totalPCMFrameCount;
    totalSamples = static_cast<int>(totalFrames * channels);

    std::cout << "   * " << sampleRate << " Hz | " << channels << " canal(es) | " << totalFrames << " frames" << std::endl;

    // Validar parametros antes de asignar memoria
    if (!validarParametrosAudio(sampleRate, channels, totalSamples)) {
        drflac_close(pFlac);
        return {};
    }

    // Leer primero a buffer temporal float
    std::vector<float> tempBuffer(totalSamples);
    drflac_uint64 samplesRead = drflac_read_pcm_frames_f32(pFlac, totalFrames, tempBuffer.data());

    drflac_close(pFlac);

    if (samplesRead != totalFrames) {
        std::cerr << "% Warning: Leidas " << samplesRead << " frames de "
            << totalFrames << " esperadas" << std::endl;
        totalSamples = static_cast<int>(samplesRead * channels);
        tempBuffer.resize(totalSamples);
    }

    // Convertir de float a AudioSample (double)
    std::vector<AudioSample> samples(totalSamples);
    
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < totalSamples; ++i) {
        samples[i] = static_cast<AudioSample>(tempBuffer[i]);
    }

    std::cout << "   & Decodificacion FLAC completada" << std::endl;
    return samples;
}

/**
 * Decodifica archivo MP3 a vector de AudioSample
 * Usa minimp3 para decodificacion
 */
static std::vector<AudioSample> decodeMp3ToVector(const char* filePath, int& sampleRate,
    int& channels, int& totalSamples) {
    std::cout << "-> Decodificando archivo MP3: " << filePath << std::endl;

    mp3dec_t mp3d;
    mp3dec_file_info_t info;
    mp3dec_init(&mp3d);

    if (mp3dec_load(&mp3d, filePath, &info, nullptr, nullptr)) {
        std::cerr << "! Error: No se pudo decodificar MP3" << std::endl;
        return {};
    }

    sampleRate = info.hz;
    channels = info.channels;
    totalSamples = static_cast<int>(info.samples);

    std::cout << "   Sample rate: " << sampleRate << " Hz" << std::endl;
    std::cout << "   Canales: " << channels << std::endl;
    std::cout << "   Muestras: " << totalSamples << std::endl;

    // Validar parametros
    if (!validarParametrosAudio(sampleRate, channels, totalSamples)) {
        free(info.buffer);
        return {};
    }

    // Convertir de int16 a AudioSample (double) con alta precision
    std::vector<AudioSample> samples(totalSamples);
    
    const double scale = 1.0 / 32768.0;  // Precision completa en double
    
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < totalSamples; ++i) {
        samples[i] = static_cast<AudioSample>(static_cast<double>(info.buffer[i]) * scale);
    }

    free(info.buffer);

    std::cout << "   & MP3 decodificado exitosamente" << std::endl;
    return samples;
}

// ============================================================================
// CONVERSION A MONO CON VECTORES
// ============================================================================

/**
 * Convierte audio multicanal a mono mediante promediado
 * Retorna nuevo vector mono (no modifica el original)
 */
static std::vector<AudioSample> convertToMono(const std::vector<AudioSample>& audio,
    int channels) {
    if (channels == 1) {
        // Ya es mono - retornar sin mensaje
        return audio;
    }

    std::cout << "\n# Conversion a mono" << std::endl;
    std::cout << "   Canales: " << channels << " -> 1 (mezclado)" << std::endl;

    const int totalSamples = static_cast<int>(audio.size());
    const int monoSamples = totalSamples / channels;

    // Validar que la division sea exacta
    if (totalSamples % channels != 0) {
        std::cerr << "% Warning: Total de muestras (" << totalSamples
            << ") no es divisible por numero de canales ("
            << channels << ")" << std::endl;
    }

    // Crear vector de salida
    std::vector<AudioSample> mono(monoSamples, 0.0);

    // Conversion paralela con precision double
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < monoSamples; ++i) {
        double sum = 0.0;  // Acumular en double
        for (int c = 0; c < channels; ++c) {
            sum += audio[i * channels + c];
        }
        mono[i] = sum / static_cast<double>(channels);
    }

    std::cout << "   @ Muestras mono: " << monoSamples << std::endl;

    return mono;
}

// ============================================================================
// CALCULAR ESTADISTICAS
// ============================================================================

/**
 * Calcula estadisticas basicas del audio
 * Util para diagnostico y validacion
 */
static void calcularEstadisticasAudio(const std::vector<AudioSample>& samples) {
    if (samples.empty()) return;

    AudioSample minVal = samples[0];
    AudioSample maxVal = samples[0];
    double sumAbs = 0.0;
    double sumSquared = 0.0;
    int zeroSamples = 0;
    int numSamples = static_cast<int>(samples.size());

    // Calcular estadisticas usando double precision
    // Nota: No usamos reduction(min/max) por conflictos con NOMINMAX en Windows
#if ENABLE_OPENMP
#pragma omp parallel for reduction(+:sumAbs,sumSquared,zeroSamples)
#endif
    for (int i = 0; i < numSamples; ++i) {
        AudioSample val = samples[i];
        
        double absVal = std::abs(val);
        sumAbs += absVal;
        sumSquared += val * val;
        
        if (absVal < 1e-12) zeroSamples++;  // Umbral mas estricto con double
    }

    // Calcular min/max despues del loop paralelo (evita problemas con NOMINMAX)
    for (int i = 0; i < numSamples; ++i) {
        AudioSample val = samples[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
    }

    AudioSample rms = std::sqrt(sumSquared / numSamples);
    AudioSample avgAbs = sumAbs / numSamples;

    std::cout << "\n# Metricas del audio" << std::endl;
    std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
    std::cout << "   | Metrica                | Umbral/Esperado        | Resultado        " << std::endl;
    std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
    std::cout << "   | Amplitud (rango)       | [-1.0, +1.0]           | [" << std::fixed << std::setprecision(4) << minVal << ", " << std::setw(6) << maxVal << "]" << std::endl;
    std::cout << "   | RMS (nivel senal)      | > 0.001                | " << std::setw(16) << rms << std::endl;
    std::cout << "   | Silencio (%)           | < 40%                  | " << std::setw(15) << (100.0 * zeroSamples / numSamples) << "%" << std::endl;
    
    // Detectar clipping
    int clipped = 0;
    for (int i = 0; i < numSamples; ++i) {
        if (std::abs(samples[i]) >= 0.99) clipped++;
    }

    if (clipped > 0) {
        double clippedPercent = 100.0 * clipped / numSamples;
        std::cout << "   | Clipping (%)           | < 5%                   | " << std::setw(15) << clippedPercent << "% |" << std::endl;
    }

    // Detectar silencio excesivo
    if (rms < 0.001) {
        std::cout << "   % Advertencia: Audio muy silencioso (RMS=" << rms << ")" << std::endl;
    }
}

// FUNCION PRINCIPAL DE DECODIFICACION 
std::vector<AudioSample> loadAudio(const char* filePath, int& sampleRate,
    int& numChannels, int& numSamples) {
    
    // Validar archivo
    if (!validarArchivo(filePath)) {
        return {};
    }

    // Extraer extension
    std::string filename(filePath);
    std::string extension;

    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) {
        std::cerr << "! Error: Archivo sin extension" << std::endl;
        return {};
    }

    extension = filename.substr(dotPos + 1);
    for (auto& c : extension) c = std::tolower(c);

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "[ETAPA 1/6] CARGA DE AUDIO" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Archivo: " << filePath << std::endl;
    std::cout << "Formato: ." << extension << std::endl;

    // Variables de decodificacion
    std::vector<AudioSample> samples;
    int totalSamples = 0;
    int channels = 0;

    // Decodificar segun formato
    if (extension == "mp3") {
        samples = decodeMp3ToVector(filePath, sampleRate, channels, totalSamples);
    }
    else if (extension == "wav" || extension == "aiff") {
        samples = decodeWavToVector(filePath, sampleRate, channels, totalSamples);
    }
    else if (extension == "flac") {
        samples = decodeFlacToVector(filePath, sampleRate, channels, totalSamples);
    }
    else {
        std::cerr << "! Error: Formato no soportado: ." << extension << std::endl;
        std::cerr << "   Formatos validos: .mp3, .wav, .aiff, .flac" << std::endl;
        return {};
    }

    // Verificar exito de decodificacion
    if (samples.empty()) {
        std::cerr << "! Error: Fallo al decodificar archivo de audio" << std::endl;
        return {};
    }

    // Validacion de integridad del audio cargado
    bool audioIntegro = true;
    int muestrasInvalidas = 0;
    
    for (size_t i = 0; i < samples.size(); ++i) {
        AudioSample val = samples[i];
        if (std::isnan(val) || std::isinf(val)) {
            muestrasInvalidas++;
            samples[i] = 0.0;  // Reemplazar valores invalidos con silencio
            audioIntegro = false;
        }
    }
    
    if (!audioIntegro) {
        std::cerr << "! Warning: " << muestrasInvalidas 
                  << " muestras invalidas detectadas y corregidas" << std::endl;
    }

    // Calcular estadisticas del audio decodificado
    calcularEstadisticasAudio(samples);

    // Convertir a mono si es necesario
    auto monoSamples = convertToMono(samples, channels);

    // Validar resultado de conversion
    if (monoSamples.empty()) {
        std::cerr << "! Error: Conversion a mono produjo vector vacio" << std::endl;
        return {};
    }

    // Ajustar salida
    numSamples = static_cast<int>(monoSamples.size());
    numChannels = 1;
    
    // VALIDAR CALIDAD DEL AUDIO (silencio, ruido, duracion)
    if (!validarCalidadAudio(monoSamples, sampleRate, numChannels, filePath)) {
        std::cerr << "! Error: El audio no cumple los criterios de calidad" << std::endl;
        return {};
    }

    // Resumen final
    numSamples = static_cast<int>(monoSamples.size());
    numChannels = 1;
    double durationSeconds = static_cast<double>(numSamples) / sampleRate;
    
    return monoSamples;  // Move semantics - sin copia
}

#include "mfcc.h"
#include "../../utils/config.h"
#include "../../utils/additional.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <vector>

// ============================================================================
// TABLA LOG PARA OPTIMIZACION (Opcional - usando double)
// ============================================================================

static std::vector<double> LOG_TABLE;
static bool LOG_TABLE_INITIALIZED = false;

static void initLogTable() {
    if (!LOG_TABLE_INITIALIZED) {
        LOG_TABLE.resize(2000);

#if ENABLE_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < 2000; ++i) {
            LOG_TABLE[i] = std::log(1e-12 + static_cast<double>(i) / 200.0);
        }

        LOG_TABLE_INITIALIZED = true;
    }
}

static inline double fastLog(double x) {
    if (x <= 1e-12) return LOG_TABLE[0];

    int index = static_cast<int>(x * 200.0);
    if (index >= 2000) return std::log(x);

    return LOG_TABLE[index];
}

// ============================================================================
// CONVERSIONES MEL-SCALE - VERSION CON DOUBLE
// ============================================================================

static inline double hzToMel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

static inline double melToHz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

// ============================================================================
// CREACION DE FILTERBANK MEL - VERSION CON VECTORES
// ============================================================================

/**
 * Crea banco de filtros mel-scale (triangulares)
 * Retorna matriz 2D [numFilters][numBins]
 */
static std::vector<std::vector<double>> createMelFilterbank(
    int sampleRate,
    int numFilters,
    int fftSize)
{
    const int numBins = fftSize / 2;

    // Obtener configuracion
    auto& cfg = CONFIG_MFCC;
    const double lowMel = hzToMel(cfg.freqMin);
    const double highMel = hzToMel(cfg.freqMax);

    if (CONFIG_PREP.verbose) {
        std::cout << "   Filterbank mel: " << numFilters << " filtros | Rango: "
            << cfg.freqMin << "-" << cfg.freqMax << " Hz" << std::endl;
    }

    // Inicializar matriz de filtros
    std::vector<std::vector<double>> filterbank(
        numFilters,
        std::vector<double>(numBins, 0.0)
    );

    // Calcular puntos mel equiespaciados
    const int numPoints = numFilters + 2;
    std::vector<double> melPoints(numPoints);
    std::vector<int> bin(numPoints);

    const double melStep = (highMel - lowMel) / (numFilters + 1);
    for (int i = 0; i < numPoints; ++i) {
        melPoints[i] = lowMel + melStep * i;
    }

    // Convertir puntos mel a bins de frecuencia
    const double binFactor = static_cast<double>(fftSize) / sampleRate;
    for (int i = 0; i < numPoints; ++i) {
        bin[i] = static_cast<int>(melToHz(melPoints[i]) * binFactor);
        if (bin[i] >= numBins) bin[i] = numBins - 1;
    }

    // Construir filtros triangulares en paralelo
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < numFilters; ++i) {
        const int start = bin[i];
        const int peak = bin[i + 1];
        const int end = bin[i + 2];

        // Rampa ascendente
        if (peak > start) {
            const double slope = 1.0 / (peak - start);
            for (int j = start; j < peak; ++j) {
                filterbank[i][j] = (j - start) * slope;
            }
        }

        // Rampa descendente
        if (end > peak) {
            const double slope = 1.0 / (end - peak);
            for (int j = peak; j < end; ++j) {
                filterbank[i][j] = (end - j) * slope;
            }
        }
    }

    return filterbank;
}

// EXTRACCION MFCC 
std::vector<std::vector<AudioSample>> extractMFCC(
    const std::vector<std::vector<AudioSample>>& spectrogram,
    int sampleRate)
{
    if (spectrogram.empty() || spectrogram[0].empty()) {
        std::cerr << "! Error: Espectrograma invalido en MFCC" << std::endl;
        return {};
    }

    // Inicializar tabla de logaritmos
    initLogTable();

    const int numFrames = static_cast<int>(spectrogram.size());
    const int numBins = static_cast<int>(spectrogram[0].size());

    // Obtener configuracion
    auto& cfg = CONFIG_MFCC;
    const int numFilters = cfg.numFilters;
    const int numCoeffs = cfg.numCoefficients;

    if (CONFIG_PREP.verbose) {
        std::cout << "-> Extrayendo MFCC" << std::endl;
        std::cout << "   Coeficientes: " << numCoeffs << " | Filtros: " << numFilters << std::endl;
    }

    // Inicializar matriz MFCC
    std::vector<std::vector<AudioSample>> mfcc(
        numFrames,
        std::vector<AudioSample>(numCoeffs, 0.0)
    );

    // Crear filterbank mel
    const int fftSize = numBins * 2;
    auto filterbank = createMelFilterbank(sampleRate, numFilters, fftSize);

    // Procesar cada frame en paralelo
#if ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int f = 0; f < numFrames; ++f) {
        // Buffer local para energias mel (privado por thread)
        std::vector<double> filterEnergies(numFilters, 0.0);

        // 1. Aplicar filterbank: calcular energia por filtro mel
        for (int i = 0; i < numFilters; ++i) {
            for (int j = 0; j < numBins; ++j) {
                filterEnergies[i] += spectrogram[f][j] * filterbank[i][j];
            }

            // Evitar log(0) con epsilon
            if (filterEnergies[i] < 1e-12) {
                filterEnergies[i] = 1e-12;
            }

            // 2. Aplicar logaritmo (compresion dinamica)
            filterEnergies[i] = std::log(filterEnergies[i]);
        }

        // 3. DCT (Discrete Cosine Transform) para decorrelacion
        for (int k = 0; k < numCoeffs; ++k) {
            double sum = 0.0;
            const double cosArg = M_PI * k / numFilters;

            for (int n = 0; n < numFilters; ++n) {
                sum += filterEnergies[n] * std::cos(cosArg * (n + 0.5));
            }

            mfcc[f][k] = sum;
        }
    }

    if (CONFIG_PREP.verbose) {
        std::cout << "   & MFCC extraido: " << numFrames << " frames x "
            << numCoeffs << " coeffs" << std::endl;
    }

    return mfcc;
}

// ============================================================================
// ESTADISTICAS MFCC 
// ============================================================================

std::vector<AudioSample> calcularEstadisticasMFCC(
    const std::vector<std::vector<AudioSample>>& mfcc)
{
    if (mfcc.empty() || mfcc[0].empty()) {
        std::cout << "% Warning: MFCC vacio, retornando vector cero" << std::endl;
        return std::vector<AudioSample>(CONFIG_MFCC.totalFeatures, 0.0);
    }

    const int frames = static_cast<int>(mfcc.size());
    const int coeffs = static_cast<int>(mfcc[0].size());

    if (CONFIG_PREP.verbose) {
        std::cout << "-> Calculando estadisticas MFCC extendidas (MEAN + STD + MIN + MAX + DELTA)" << std::endl;
    }

    // Vectores para cada estadistica
    std::vector<AudioSample> medias(coeffs, 0.0);
    std::vector<AudioSample> desviaciones(coeffs, 0.0);
    std::vector<AudioSample> minimos(coeffs, std::numeric_limits<AudioSample>::max());
    std::vector<AudioSample> maximos(coeffs, std::numeric_limits<AudioSample>::lowest());
    std::vector<AudioSample> deltas(coeffs, 0.0);

    // PASO 1: Calcular MEAN, MIN, MAX en paralelo
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int c = 0; c < coeffs; ++c) {
        double suma = 0.0;
        AudioSample minVal = std::numeric_limits<AudioSample>::max();
        AudioSample maxVal = std::numeric_limits<AudioSample>::lowest();
        
        for (int f = 0; f < frames; ++f) {
            AudioSample val = mfcc[f][c];
            suma += val;
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
        
        medias[c] = suma / frames;
        minimos[c] = minVal;
        maximos[c] = maxVal;
    }

    // PASO 2: Calcular STD (desviacion estandar)
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int c = 0; c < coeffs; ++c) {
        double sumaVarianza = 0.0;
        for (int f = 0; f < frames; ++f) {
            double diff = mfcc[f][c] - medias[c];
            sumaVarianza += diff * diff;
        }
        desviaciones[c] = std::sqrt(sumaVarianza / frames);
    }

    // PASO 3: Calcular DELTA (primera derivada temporal - promedio de cambios)
    // Delta captura la dinamica temporal de la voz
#if ENABLE_OPENMP
#pragma omp parallel for
#endif
    for (int c = 0; c < coeffs; ++c) {
        double sumaDelta = 0.0;
        for (int f = 1; f < frames; ++f) {
            sumaDelta += (mfcc[f][c] - mfcc[f-1][c]);
        }
        deltas[c] = (frames > 1) ? (sumaDelta / (frames - 1)) : 0.0;
    }

    // PASO 4: Concatenar todas las estadisticas
    // Orden: [MEAN(50), STD(50), MIN(50), MAX(50), DELTA(50)] = 250 features
    std::vector<AudioSample> features;
    features.reserve(coeffs * 5);
    
    // Agregar MEAN
    features.insert(features.end(), medias.begin(), medias.end());
    // Agregar STD
    features.insert(features.end(), desviaciones.begin(), desviaciones.end());
    // Agregar MIN
    features.insert(features.end(), minimos.begin(), minimos.end());
    // Agregar MAX
    features.insert(features.end(), maximos.begin(), maximos.end());
    // Agregar DELTA
    features.insert(features.end(), deltas.begin(), deltas.end());

    if (CONFIG_PREP.verbose) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "[ETAPA 5/6] CALCULO DE ESTADISTICAS TEMPORALES" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\n# Estadisticas calculadas sobre " << frames << " frames" << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Estadistica            | Coeficientes           | Dimension        " << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | MEAN (media temporal)  | " << std::setw(22) << coeffs << " | " << std::setw(10) << coeffs << " feat. " << std::endl;
        std::cout << "   | STD (desv. estandar)   | " << std::setw(22) << coeffs << " | " << std::setw(10) << coeffs << " feat. " << std::endl;
        std::cout << "   | MIN (valor minimo)     | " << std::setw(22) << coeffs << " | " << std::setw(10) << coeffs << " feat. " << std::endl;
        std::cout << "   | MAX (valor maximo)     | " << std::setw(22) << coeffs << " | " << std::setw(10) << coeffs << " feat. " << std::endl;
        std::cout << "   | DELTA (primera deriv.) | " << std::setw(22) << coeffs << " | " << std::setw(10) << coeffs << " feat. " << std::endl;
        std::cout << "\n@ Total features extraidos: " << features.size() << " (" << coeffs << " x 5 estadisticas)" << std::endl;
    }

    return features;
}

// ============================================================================
// I/O DE CARACTERISTICAS - VERSION CON DOUBLE
// ============================================================================

bool guardarCaracteristicasBinario(
    const std::vector<AudioSample>& features,
    int label,
    const std::string& outputPath)
{
    std::ofstream out(outputPath, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo abrir " << outputPath << std::endl;
        return false;
    }

    const int dim = static_cast<int>(features.size());

    // Formato: [dim:int][features:double*dim][label:int]
    out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
    out.write(reinterpret_cast<const char*>(features.data()), sizeof(AudioSample) * dim);
    out.write(reinterpret_cast<const char*>(&label), sizeof(int));

    out.close();
    return true;
}

bool exportarCaracteristicasCSV(
    const std::vector<std::vector<AudioSample>>& features,
    const std::vector<int>& labels,
    const std::string& outputPath)
{
    std::ofstream out(outputPath);
    if (!out.is_open()) {
        std::cerr << "! Error: No se pudo crear " << outputPath << std::endl;
        return false;
    }

    if (CONFIG_PREP.verbose) {
        std::cout << "-> Exportando a CSV: " << outputPath << std::endl;
    }

    // Header CSV
    out << "label";
    if (!features.empty() && !features[0].empty()) {
        for (size_t i = 0; i < features[0].size(); ++i) {
            out << ",mfcc" << i;
        }
    }
    out << "\n";

    // Datos
    for (size_t i = 0; i < features.size(); ++i) {
        out << labels[i];
        for (AudioSample feature : features[i]) {
            out << "," << std::fixed << std::setprecision(12) << feature;
        }
        out << "\n";
    }

    out.close();

    if (CONFIG_PREP.verbose) {
        std::cout << "   & CSV exportado: " << features.size() << " muestras" << std::endl;
    }

    return true;
}
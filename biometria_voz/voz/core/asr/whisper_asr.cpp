#include "whisper_asr.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <regex>
#include <cstdio>
#include "similaridad.h"
#include <iomanip> 
// Cache thread-local
namespace {
    thread_local std::string _ultimaRutaAudio;
    thread_local std::string _ultimaTranscripcion;
}

// Función auxiliar para transcribir y cachear resultado
static std::string transcribirYCached(const std::string& audioPath) {
    if (audioPath == _ultimaRutaAudio && !_ultimaTranscripcion.empty()) {
        return _ultimaTranscripcion;
    }
    
    // Detectar ejecutable según sistema operativo
    #ifdef _WIN32
        std::string whisperExe = ".\\whisper-cli.exe";
    #else
        std::string whisperExe = "./whisper-cli";
    #endif
    
    std::string comando =
        whisperExe + " "
        "-m ggml-tiny.bin "
        "-f \"" + audioPath + "\" "
        "--language es "
        "--no-timestamps --no-prints 2>&1";  // Suprimir logs de debug

    std::cout << "-> Ejecutando Whisper: " << comando << std::endl;
    
    // Usar popen para capturar salida directamente
    #ifdef _WIN32
        FILE* pipe = _popen(comando.c_str(), "r");
    #else
        FILE* pipe = popen(comando.c_str(), "r");
    #endif
    
    if (!pipe) {
        std::cerr << "Error al ejecutar whisper.\n";
        _ultimaTranscripcion.clear();
        return "";
    }
    
    // Leer salida directamente
    std::stringstream buffer;
    char linea[256];
    while (fgets(linea, sizeof(linea), pipe) != nullptr) {
        buffer << linea;
    }
    
    #ifdef _WIN32
        int result = _pclose(pipe);
    #else
        int result = pclose(pipe);
    #endif
    
    if (result != 0) {
        std::cerr << "Whisper termino con error: " << result << "\n";
    }

    _ultimaRutaAudio = audioPath;
    _ultimaTranscripcion = buffer.str();
    return _ultimaTranscripcion;
}

// Normaliza texto
std::string normalizarTxt(const std::string& texto) {
    std::string limpio;
    for (char c : texto) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalpha(uc) || std::isspace(uc)) {
            limpio += std::tolower(uc);
        }
        else if (std::isdigit(uc)) {
            limpio += c;
        }
    }
    std::regex multiEspacio("\\s+");
    limpio = std::regex_replace(limpio, multiEspacio, " ");
    if (!limpio.empty() && limpio.front() == ' ') limpio.erase(0, 1);
    if (!limpio.empty() && limpio.back() == ' ') limpio.pop_back();
    return limpio;
}

// API pública: compara contra la frase estática
bool transcribeAndCompare(const std::string& audioPath, const std::string& fraseEsperada) {
    std::string transcripcion = transcribirYCached(audioPath);
    if (transcripcion.empty()) return false;

    std::string esperado = normalizarTxt(fraseEsperada);
    std::string detectado = normalizarTxt(transcripcion);

    double similitud = porcentajeSimilitud(esperado, detectado);
    std::cout << "Transcripción detectada: " << detectado << "\n";
    std::cout << "Frase esperada: " << esperado << "\n";
    std::cout << "Similitud: " << (similitud * 100) << "%\n";
    return similitud >= 0.85;
}

// API pública: devuelve transcripción normalizada (si la quieres usar)
std::string obtenerTranscripcion(const std::string& audioPath) {
    std::string transcripcion = transcribirYCached(audioPath);
    return normalizarTxt(transcripcion);
}

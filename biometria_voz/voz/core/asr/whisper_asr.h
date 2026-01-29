#ifndef WHISPER_ASR_H
#define WHISPER_ASR_H

#include <string>

bool transcribeAndCompare(const std::string& audioPath, const std::string& fraseEsperada);
std::string obtenerTranscripcion(const std::string& audioPath);
std::string normalizarTxt(const std::string& texto);

#endif

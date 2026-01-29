#pragma once

#include "httplib.h"
#include "json.hpp"

#include <string>
#include <cstddef>

using json = nlohmann::json;

// Helpers de texto / JSON (movidos desde servidor.cpp)
std::string trunc(const std::string& s, size_t n = 120);
std::string truncN(const std::string& s, size_t max_chars = 1500);
std::string jsonCampo(const json& j, const char* k);

std::string repLine(char ch = '=', int n = 60);
std::string repKV(const std::string& k, const std::string& v);
std::string repQC(const std::string& metrica, const std::string& valor,
                  const std::string& umbral, const std::string& detalle = "");

std::string resumenUsuarioJson(const json& datos);

// Validaciones simples (movidas desde servidor.cpp)
bool esEntero(const std::string& s);
bool esDoubleSimple(const std::string& s);

// Params HTTP seguros (movidos desde servidor.cpp)
std::string safeParam(const httplib::Request& req, const std::string& name);

#pragma once

#include "json.hpp"
#include <string>

using json = nlohmann::json;

struct ExitMapped {
    int http_status = 500;
    std::string title;     // categoría corta
    std::string message;   // explicación técnica
};

// Mapea (proc, exit_code) -> http + mensaje.
// IMPORTANTE: no asume significados específicos si no están definidos.
ExitMapped mapExitCode(const std::string& proc, int exit_code);

json buildPublicErrorBody(int http_status, const std::string& title, const std::string& message);

// Construye un JSON de error estándar (para respuestas coherentes)
json buildErrorBody(const std::string& proc, int exit_code, const ExitMapped& mapped, const std::string& stderr_tail = "", const std::string& stdout_tail = "");


#include "proc_utils.h"

#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#else
  #error "Se requiere <filesystem> (C++17+)."
#endif

#ifndef _WIN32
    #include <sys/wait.h>
#endif

int systemExitCode(int status) {
#if defined(_WIN32)
    return status;
#else
    if (status == -1) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 99;
#endif
}

// ===================== Helpers (para logs útiles) =====================
std::string leerUltimasLineas(const std::string& path, size_t max_lines) {
    std::ifstream f(path);
    if (!f.is_open()) return "";

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        lines.push_back(line);
        if (lines.size() > max_lines) {
            lines.erase(lines.begin());
        }
    }

    std::ostringstream oss;
    for (const auto& l : lines) oss << l << "\n";
    return oss.str();
}

// Carpeta tmp consistente (mejor en /tmp dentro del contenedor)
void limpiarDirectorio(const std::string& ruta) {
    try {
        if (fs::exists(ruta)) fs::remove_all(ruta);
    } catch (const std::exception& e) {
        std::cerr << "Error limpiando directorio: " << e.what() << "\n";
    }
}

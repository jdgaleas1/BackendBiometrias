#pragma once

#include <string>
#include <cstddef>

// Normaliza el status devuelto por system() u otra llamada similar
int systemExitCode(int status);

// Lee las últimas N líneas de un archivo (útil para stderr/out)
std::string leerUltimasLineas(const std::string& path, size_t max_lines = 40);

// Limpia (borra) el directorio indicado si existe
void limpiarDirectorio(const std::string& ruta);

#ifndef CARGAR_IMAGEN_H
#define CARGAR_IMAGEN_H

#include <string>
#include <cstdint>

// Carga una imagen desde el disco y devuelve un puntero a los datos
unsigned char* cargarImagen(const std::string& ruta, int& ancho, int& alto, int& canales, int forzarCanales = 0);
void liberarImagen(unsigned char* datos);

#endif

// cargar_imagen.cpp

#define STB_IMAGE_IMPLEMENTATION
#include "../extern/stb_image.h"
#include "cargar_imagen.h"
#include <iostream>

using namespace std;

unsigned char* cargarImagen(const string& ruta, int& ancho, int& alto, int& canales, int forzarCanales)
{
    unsigned char* datos = stbi_load(ruta.c_str(), &ancho, &alto, &canales, forzarCanales);
    if (!datos) {
        cerr << "Error al cargar la imagen: " << ruta << endl;
        return nullptr;
    }
    return datos;
}

void liberarImagen(unsigned char* datos)
{
    if (datos) stbi_image_free(datos);
}
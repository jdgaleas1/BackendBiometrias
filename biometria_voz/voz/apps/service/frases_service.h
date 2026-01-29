#pragma once
#include <string>
#include <vector>
#include "../../external/json.hpp"

class FrasesService {
public:
    nlohmann::json insertarFrase(const std::string& frase);
    nlohmann::json obtenerTodas();
    nlohmann::json obtenerFraseAleatoria();
    nlohmann::json actualizarEstadoFrase(int id_texto, int activo);
    nlohmann::json obtenerFrasePorId(int id);
    nlohmann::json eliminarFrase(int id_texto);
};

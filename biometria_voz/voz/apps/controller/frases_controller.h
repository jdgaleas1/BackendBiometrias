#pragma once
#include <string>
#include <vector>
#include "../service/frases_service.h"
#include "../../external/json.hpp"

class FrasesController {
public:
    FrasesController();
    nlohmann::json agregarFrase(const std::string& frase);
    nlohmann::json listarFrases();
    nlohmann::json obtenerFraseAleatoria();
    nlohmann::json actualizarEstadoFrase(int id_texto, int activo);
    nlohmann::json obtenerFrasePorId(int id);
    nlohmann::json eliminarFrase(int id_texto);
private:
    FrasesService service;
};

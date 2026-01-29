#include "frases_service.h"
#include "../../utils/http_helpers.h"
#include <random>
#include <chrono>
#include <iostream>

using namespace HttpHelpers;

nlohmann::json FrasesService::insertarFrase(const std::string& frase) {
    nlohmann::json body;
    body["frase"] = frase;

    auto res = hacerPOST("/textos_dinamicos_audio", body, 15);

    nlohmann::json response;
    nlohmann::json output;
    
    if (procesarResponsePOST(res, output)) {
        response["success"] = true;
        response["message"] = "Frase agregada";
        response["data"] = output;
    } else {
        response["success"] = false;
        response["message"] = "Error al agregar frase";
        if (res) {
            response["debug_status"] = res->status;
            response["debug_body"] = res->body;
        } else {
            response["debug"] = "No se pudo conectar a PostgREST";
        }
    }

    return response;
}

nlohmann::json FrasesService::obtenerTodas() {
    auto res = hacerGET("/textos_dinamicos_audio", 15);

    nlohmann::json frases;
    if (procesarResponseGET(res, frases)) {
        nlohmann::json response;
        response["success"] = true;
        response["frases"] = frases;
        response["total"] = frases.size();
        return response;
    }

    nlohmann::json error;
    error["success"] = false;
    error["error"] = "Error al obtener frases";
    return error;
}

nlohmann::json FrasesService::obtenerFraseAleatoria() {
    // Obtener todas las frases activas
    auto res = hacerGET("/textos_dinamicos_audio?estado_texto=eq.activo", 15);

    nlohmann::json todasFrases;
    if (!procesarResponseGET(res, todasFrases)) {
        nlohmann::json error;
        error["success"] = false;
        error["error"] = "Error al obtener frases";
        return error;
    }

    // Filtrar frases que NO hayan alcanzado el limite de usos
    nlohmann::json frasesDisponibles = nlohmann::json::array();
    for (const auto& frase : todasFrases) {
        int contadorActual = frase.value("contador_usos", 0);
        int limiteUsos = frase.value("limite_usos", 150);
        
        if (contadorActual < limiteUsos) {
            frasesDisponibles.push_back(frase);
        }
    }

    if (frasesDisponibles.empty()) {
        nlohmann::json error;
        error["success"] = false;
        error["error"] = "No hay frases disponibles (todas alcanzaron el limite de usos)";
        return error;
    }

    // Generador aleatorio
    std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<> dis(0, frasesDisponibles.size() - 1);
    int index = dis(gen);

    int idTexto = frasesDisponibles[index]["id_texto"];
    int contadorActual = frasesDisponibles[index].value("contador_usos", 0);
    int limiteUsos = frasesDisponibles[index].value("limite_usos", 150);

    // Incrementar contador de usos
    nlohmann::json updateBody;
    updateBody["contador_usos"] = contadorActual + 1;
    
    // Si alcanza el limite, desactivar automaticamente
    if ((contadorActual + 1) >= limiteUsos) {
        updateBody["estado_texto"] = "desactivado";
        std::cout << "# Frase ID " << idTexto << " desactivada automaticamente (limite alcanzado)" << std::endl;
    }

    std::string urlUpdate = "/textos_dinamicos_audio?id_texto=eq." + std::to_string(idTexto);
    auto resUpdate = hacerPATCH(urlUpdate, updateBody, 15);

    if (!procesarResponseNoContent(resUpdate)) {
        std::cerr << "! Warning: No se pudo actualizar contador de frase ID " << idTexto << std::endl;
    }

    nlohmann::json response;
    response["success"] = true;
    response["frase"] = frasesDisponibles[index]["frase"];
    response["id_texto"] = idTexto;
    response["contador_usos"] = contadorActual + 1;
    response["limite_usos"] = limiteUsos;
    return response;
}

nlohmann::json FrasesService::actualizarEstadoFrase(int id_texto, int activo) {
    nlohmann::json body;
    body["estado_texto"] = (activo == 1) ? "activo" : "desactivado";

    std::string url = "/textos_dinamicos_audio?id_texto=eq." + std::to_string(id_texto);
    auto res = hacerPATCH(url, body, 15);

    nlohmann::json response;
    if (procesarResponseNoContent(res)) {
        response["success"] = true;
        response["message"] = (activo == 1) ? "Frase activada" : "Frase desactivada";
    } else {
        response["success"] = false;
        response["message"] = "Error al actualizar estado";
    }

    return response;
}

nlohmann::json FrasesService::obtenerFrasePorId(int id) {
    std::cout << "   [DEBUG] FrasesService::obtenerFrasePorId - ID: " << id << std::endl;
    
    std::string endpoint = "/textos_dinamicos_audio?id_texto=eq." + std::to_string(id);
    std::cout << "   [DEBUG] Endpoint: " << endpoint << std::endl;
    
    auto res = hacerGET(endpoint, 15);
    
    nlohmann::json frases;
    if (!procesarResponseGET(res, frases)) {
        std::cerr << "   ! ERROR: No se pudo obtener frase" << std::endl;
        nlohmann::json error;
        error["success"] = false;
        error["error"] = "No se pudo conectar a PostgREST";
        return error;
    }

    std::cout << "   [DEBUG] Frases encontradas: " << frases.size() << std::endl;
    
    if (!frases.empty()) {
        std::cout << "   @ Frase obtenida exitosamente" << std::endl;
        return frases[0]; // Retorna la primera (y única) frase
    }

    std::cout << "   ! Frase no encontrada con ID " << id << std::endl;
    nlohmann::json error;
    error["success"] = false;
    error["error"] = "Frase no encontrada";
    return error;
}

nlohmann::json FrasesService::eliminarFrase(int id_texto) {
    std::string url = "/textos_dinamicos_audio?id_texto=eq." + std::to_string(id_texto);
    auto res = hacerDELETE(url, 15);

    nlohmann::json response;
    if (procesarResponseNoContent(res)) {
        response["success"] = true;
        response["message"] = "Frase eliminada correctamente";
    } else {
        response["success"] = false;
        response["message"] = "Error al eliminar frase";
        if (res) {
            response["debug_status"] = res->status;
            response["debug_body"] = res->body;
        }
    }

    return response;
}
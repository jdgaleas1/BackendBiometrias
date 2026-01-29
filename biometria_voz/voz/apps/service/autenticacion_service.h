#ifndef AUTENTICACION_SERVICE_H
#define AUTENTICACION_SERVICE_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include "../../utils/config.h"
#include "../../core/classification/svm.h"
#include "../service/frases_service.h"
struct ResultadoAutenticacion {
    bool exito;
    bool autenticado;
    int userId;
    std::string userName;
    double confianza;
    int tiempoProcesamiento;
    std::map<int, double> scores;
    std::string error;

    // verificación de texto dinámico
    std::string fraseEsperada;
    std::string transcripcionDetectada;
    double similitudTexto;
    bool textoCoincide;
};


class AutenticacionService {
private:
    ModeloSVM modelo;
    std::map<int, std::string> mapeoUsuarios;
    FrasesService frasesService;

public:
    AutenticacionService(const std::string& modelPath, const std::string& mappingPath);

    ResultadoAutenticacion autenticar(const std::string& audioPath, const std::string& identificador = "", int idFrase = -1);
    void recargarModelo(const std::string& modelPath);
    void recargarMapeos(const std::string& mappingPath);

private:
    bool procesarAudio(const std::string& audioPath, std::vector<AudioSample>& features);
};

#endif
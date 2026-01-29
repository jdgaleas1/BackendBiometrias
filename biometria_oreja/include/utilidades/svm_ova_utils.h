#ifndef SVM_OVA_UTILS_H
#define SVM_OVA_UTILS_H

#include "svm/svm_entrenamiento.h"

#include <string>
#include <vector>
#include <cstdint>

void dilatacion3x3_binaria(uint8_t* data, int ancho, int alto);

bool cargarModeloSVM(const std::string& ruta, ModeloSVM& modelo);

bool guardarModeloSVM(const std::string& ruta, const ModeloSVM& modelo);

#endif

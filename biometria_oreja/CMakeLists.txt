cmake_minimum_required(VERSION 3.14)
project(biometria_oreja)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
include_directories(extern)

find_package(OpenMP REQUIRED)

# =========================================================
# 1) oreja_core (runtime / inferencia)
# =========================================================
file(GLOB_RECURSE SRC_PREPROC CONFIGURE_DEPENDS
    src/preprocesamiento/*.cpp
)

file(GLOB_RECURSE SRC_FEATURES CONFIGURE_DEPENDS
    src/extraccion_caracteristicas/*.cpp
)

set(SRC_SVM_INFER
    src/svm/svm_prediccion.cpp
    src/svm/cargar_csv.cpp
)

set(SRC_UTIL_CORE
    src/utilidades/normalizacion.cpp
    src/utilidades/pca_utils.cpp
    src/utilidades/lda_utils.cpp
    src/utilidades/logger.cpp
    src/utilidades/guardar_pgm.cpp
    src/utilidades/svm_ova_utils.cpp
    src/cargar_imagen.cpp
    src/utilidades/zscore_params.cpp
)

add_library(oreja_core STATIC
    ${SRC_PREPROC}
    ${SRC_FEATURES}
    ${SRC_SVM_INFER}
    ${SRC_UTIL_CORE}
)

target_include_directories(oreja_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(oreja_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(NOT WIN32)
    target_link_libraries(oreja_core PUBLIC pthread)
endif()

# =========================================================
# 2) oreja_train (offline: entrenamiento + métricas + splits)
# =========================================================
set(SRC_SVM_TRAIN
    src/svm/svm_entrenamiento.cpp
)

file(GLOB_RECURSE SRC_METRICAS CONFIGURE_DEPENDS
    src/metricas/*.cpp
)

set(SRC_UTIL_TRAIN
    src/utilidades/dividir_dataset.cpp
    src/utilidades/dataset_loader.cpp
    src/utilidades/guardar_csv.cpp
)

add_library(oreja_train STATIC
    ${SRC_SVM_TRAIN}
    ${SRC_METRICAS}
    ${SRC_UTIL_TRAIN}
)

target_include_directories(oreja_train PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern
)

target_link_libraries(oreja_train PUBLIC oreja_core)

# =========================================================
# 3) oreja_admin (online admin: registro + reentreno incremental)
#    AHORA compila src/admin/*.cpp
# =========================================================
file(GLOB_RECURSE SRC_ADMIN CONFIGURE_DEPENDS
    src/admin/*.cpp
)

add_library(oreja_admin STATIC
    ${SRC_ADMIN}
)

target_include_directories(oreja_admin PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern
)

# oreja_admin usa core + train
target_link_libraries(oreja_admin PUBLIC oreja_core oreja_train)

# =========================================================
# 4) Ejecutables
# =========================================================
add_executable(procesar_imagen apps/visualizacion/procesar_imagen.cpp)
target_link_libraries(procesar_imagen PRIVATE oreja_core)

add_executable(predecir apps/produccion/predecir.cpp)
target_link_libraries(predecir PRIVATE oreja_core)

add_executable(evaluar_batch_qc apps/produccion/evaluar_batch_qc.cpp)
target_link_libraries(evaluar_batch_qc PRIVATE oreja_core)

add_executable(procesar_dataset apps/entrenamiento/procesar_dataset.cpp)
target_compile_definitions(procesar_dataset PRIVATE PREPROCESO_SEQ)
target_link_libraries(procesar_dataset PRIVATE oreja_train)

add_executable(procesar_dataset_secuencial apps/entrenamiento/procesar_dataset_secuencial.cpp)
target_link_libraries(procesar_dataset_secuencial PRIVATE oreja_train)

add_executable(entrenar_svm apps/entrenamiento/entrenar_svm.cpp)
target_link_libraries(entrenar_svm PRIVATE oreja_train)

add_executable(curva_aprendizaje apps/visualizacion/curva_aprendizaje.cpp)
target_link_libraries(curva_aprendizaje PRIVATE oreja_train)

add_executable(calcular_umbrales_1_1 apps/entrenamiento/calcular_umbrales_1_1.cpp)
target_link_libraries(calcular_umbrales_1_1 PRIVATE oreja_train)

add_executable(evaluar_metricas_tecnicas apps/analisis/evaluar_metricas_tecnicas.cpp)
target_link_libraries(evaluar_metricas_tecnicas PRIVATE oreja_train)

add_executable(agregar_usuario_biometria apps/gestor_usuarios/agregar_usuario_biometria.cpp)
target_link_libraries(agregar_usuario_biometria PRIVATE oreja_admin)

add_executable(agregar_usuario apps/gestor_usuarios/agregar_usuario.cpp)
target_link_libraries(agregar_usuario PRIVATE oreja_admin)

add_executable(reentrenar_svm apps/gestor_usuarios/reentrenar_svm_incremental.cpp)
target_link_libraries(reentrenar_svm PRIVATE oreja_admin)

add_executable(servidor
    apps/servidor.cpp
    apps/server_utils.cpp
    apps/server_env.cpp
    apps/proc_utils.cpp
    apps/exit_map.cpp
    apps/report_format.cpp
)
target_link_libraries(servidor PRIVATE oreja_admin)

# =========================================================
# 5) Windows libs
# =========================================================
if (WIN32)
    target_link_libraries(procesar_imagen PRIVATE psapi)
    target_link_libraries(procesar_dataset PRIVATE psapi)
    target_link_libraries(entrenar_svm PRIVATE psapi)
    target_link_libraries(predecir PRIVATE psapi)
    target_link_libraries(evaluar_batch_qc PRIVATE psapi)
    target_link_libraries(evaluar_metricas_tecnicas PRIVATE psapi)
    target_link_libraries(agregar_usuario PRIVATE psapi)
    target_link_libraries(reentrenar_svm PRIVATE psapi)
endif()

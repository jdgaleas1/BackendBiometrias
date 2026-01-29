cmake_minimum_required(VERSION 3.15)
project(voz)
set(CMAKE_CXX_STANDARD 20)

# Detectar VOZ_ROOT correctamente - CORREGIDO
set(VOZ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# Agregar TODOS los directorios de include necesarios GLOBALMENTE
include_directories(
    ${VOZ_ROOT}
    ${VOZ_ROOT}/core
    ${VOZ_ROOT}/core/preprocessing
    ${VOZ_ROOT}/core/features
    ${VOZ_ROOT}/core/segmentation
    ${VOZ_ROOT}/core/classification
    ${VOZ_ROOT}/core/process_dataset
    ${VOZ_ROOT}/core/load_audio
    ${VOZ_ROOT}/core/augmentation
    ${VOZ_ROOT}/core/pipeline
    ${VOZ_ROOT}/core/asr
    ${VOZ_ROOT}/utils
    ${VOZ_ROOT}/apps
    ${VOZ_ROOT}/apps/service
    ${VOZ_ROOT}/apps/controller
    ${VOZ_ROOT}/apps/testeo
    ${VOZ_ROOT}/apps/testeo/verificaciones
    ${VOZ_ROOT}/apps/testeo/exportaciones
    ${VOZ_ROOT}/apps/testeo/asr-docker
    ${VOZ_ROOT}/external
)

# ARCHIVOS FUENTE 
# Archivos core
file(GLOB_RECURSE SRC_CORE CONFIGURE_DEPENDS
    "core/preprocessing/*.cpp" 
    "core/features/*.cpp"
    "core/segmentation/*.cpp" 
    "core/classification/*.cpp"
    "core/process_dataset/*.cpp"
    "core/load_audio/*.cpp" 
    "core/augmentation/*.cpp"
    "core/pipeline/*.cpp"
    "core/asr/*.cpp"
)
 
# Archivos utils 
file(GLOB_RECURSE SRC_UTILS CONFIGURE_DEPENDS
    "utils/*.cpp"
)

# Archivos services y controllers
file(GLOB_RECURSE SRC_SERVICES CONFIGURE_DEPENDS
    "apps/service/*.cpp" 
    "apps/controller/*.cpp"
)

set(SRC_COMMON ${SRC_CORE} ${SRC_UTILS})
set(SRC_FULL ${SRC_CORE} ${SRC_UTILS} ${SRC_SERVICES})

message(STATUS "-> Archivos core detectados: ${SRC_CORE}")
message(STATUS "-> Archivos utils detectados: ${SRC_UTILS}")
message(STATUS "-> Archivos services detectados: ${SRC_SERVICES}")

# DEFINICION DE EJECUTABLES CON PREFIJO voz
# Grupo 1: Procesamiento de datasets
add_executable(voz_procesar_dataset "apps/testeo/procesar_dataset.cpp" ${SRC_COMMON})
add_executable(voz_verificar_dataset "apps/testeo/verificaciones/verificar_dataset.cpp" ${SRC_FULL})
add_executable(voz_exportar_audios "apps/testeo/exportaciones/exportar_audios.cpp" ${SRC_COMMON})
add_executable(voz_exportar_features "apps/testeo/exportaciones/exportar_features.cpp" ${SRC_COMMON})

# Grupo 2: Entrenamiento y validacion
add_executable(voz_entrenar_modelo "apps/testeo/entrenar_modelo.cpp" ${SRC_CORE})
add_executable(voz_verificar_modelo "apps/testeo/verificaciones/verificar_modelo.cpp" ${SRC_FULL})
add_executable(voz_verificar_disjucion "apps/testeo/verificaciones/virificar_disjucion.cpp" ${SRC_CORE})

# Grupo 3: Testing de componentes
add_executable(voz_test_asr "apps/testeo/asr-docker/test_asr.cpp" ${SRC_CORE})
add_executable(voz_evaluar_modelo "apps/testeo/evaluarModelo.cpp" ${SRC_COMMON})

# Grupo 4: Servidores
add_executable(voz_servidor_biometrico "apps/servidor_biometrico.cpp" ${SRC_FULL})
add_executable(voz_asr_server "apps/testeo/asr-docker/asr_server.cpp" ${SRC_CORE})
add_executable(voz_api_postgres "apps/testeo/asr-docker/apiPostgres.cpp" ${SRC_CORE})

set(ALL_TARGETS 
    voz_procesar_dataset
    voz_verificar_dataset
    voz_exportar_audios
    voz_exportar_features
    voz_entrenar_modelo
    voz_verificar_modelo
    voz_verificar_disjucion
    voz_test_asr
    voz_evaluar_modelo
    voz_servidor_biometrico
    voz_asr_server
    voz_api_postgres
)

# OPENMP (PARALELIZACION)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    foreach(target ${ALL_TARGETS})
        if(TARGET ${target})
            target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endforeach()
    
    # Detectar version de OpenMP
    if(OpenMP_CXX_VERSION)
        message(STATUS "-> OpenMP habilitado: version ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "-> OpenMP habilitado para procesamiento paralelo")
    endif()
else()
    message(WARNING "% OpenMP no encontrado - compilacion sin paralelizacion")
endif()

# LIBANL (REQUERIDA POR HTTPLIB EN LINUX)
set(TARGETS_WITH_HTTPLIB
    voz_api_postgres
    voz_asr_server
    voz_servidor_biometrico
    voz_verificar_modelo
    voz_verificar_dataset
)

if(UNIX AND NOT APPLE)
    foreach(target ${TARGETS_WITH_HTTPLIB})
        if(TARGET ${target})
            target_link_libraries(${target} PRIVATE anl)
        endif()
    endforeach()
    message(STATUS "-> Biblioteca libanl vinculada para targets con httplib")
endif()

# ARCHIVOS ADICIONALES (MODELO WHISPER, DIRECTORIOS)
set(WHISPER_DEST "${CMAKE_CURRENT_BINARY_DIR}")
set(MODEL_SOURCE "${VOZ_ROOT}/core/asr/models/ggml-tiny.bin")

# Solo copiar el modelo
if(EXISTS "${MODEL_SOURCE}")
    configure_file("${MODEL_SOURCE}" "${WHISPER_DEST}/ggml-tiny.bin" COPYONLY)
    message(STATUS "-> Modelo Whisper copiado")
else()
    message(WARNING "% Modelo Whisper NO encontrado en: ${MODEL_SOURCE}")
endif()

# ============================================================================
# LIBRERIA NATIVA PARA APLICACIONES MOVILES (FFI) - OPCIONAL
# ============================================================================

option(BUILD_MOBILE_LIB "Compilar libreria movil para Flutter (requiere SQLite con -fPIC)" OFF)

if(BUILD_MOBILE_LIB)
    message(STATUS "-> Compilando libreria movil (libvoz_mobile.so)")
    
    # Crear target SQLite3 desde archivos amalgamation locales
    add_library(sqlite3_local STATIC "${CMAKE_CURRENT_SOURCE_DIR}/external/sqlite3.c")
    target_include_directories(sqlite3_local PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/external")
    target_compile_definitions(sqlite3_local PRIVATE 
        SQLITE_ENABLE_COLUMN_METADATA
        SQLITE_ENABLE_FTS5
        SQLITE_ENABLE_RTREE
    )
    
    # CRITICO: SQLite3 necesita -fPIC para enlazarse en .so
    set_target_properties(sqlite3_local PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

# Archivos de la API movil
set(MOBILE_API_SOURCES
    "apps/mobile/mobile_api.cpp"
    "apps/mobile/sqlite_adapter.cpp"
)

# Crear libreria compartida (.so / .dylib / .dll)
add_library(voz_mobile SHARED 
    ${MOBILE_API_SOURCES}
    ${SRC_CORE}
    ${SRC_UTILS}
)

# Vincular SQLite local
target_link_libraries(voz_mobile PRIVATE sqlite3_local)
    
    # POLITICA: voz_mobile SIEMPRE sin OpenMP (ejecucion secuencial)
    # Razon: Evita problemas de runtime en Android/iOS, mejor portabilidad
    # Los pragmas #pragma omp se ignoran automaticamente (ejecucion secuencial)
    message(STATUS "   @ voz_mobile: Compilacion SECUENCIAL (sin OpenMP)")
    message(STATUS "      -> Mejor portabilidad y compatibilidad movil")
    
    # Configuracion de compilacion
    set_target_properties(voz_mobile PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
        VERSION 1.0.0
        SOVERSION 1
    )
    
    # Configuracion especifica por plataforma
    if(ANDROID)
        # Android: generar libvoz_mobile.so para ARM/ARM64
        set_target_properties(voz_mobile PROPERTIES
            LIBRARY_OUTPUT_NAME "voz_mobile"
            PREFIX "lib"
            SUFFIX ".so"
        )
    elseif(APPLE)
        # iOS/macOS: generar libvoz_mobile.dylib
        set_target_properties(voz_mobile PROPERTIES
            LIBRARY_OUTPUT_NAME "voz_mobile"
            PREFIX "lib"
            SUFFIX ".dylib"
        )
    else()
        # Linux/Windows: generar libvoz_mobile.so / voz_mobile.dll
        set_target_properties(voz_mobile PROPERTIES
            LIBRARY_OUTPUT_NAME "voz_mobile"
        )
    endif()
    
    # Instalar libreria (opcional)
    install(TARGETS voz_mobile
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    # Instalar headers
    install(FILES "apps/mobile/mobile_api.h" DESTINATION include)
    
    message(STATUS "   @ Libreria: libvoz_mobile (shared library)")
    message(STATUS "   @ SQLite3 amalgamation vinculada correctamente")
else()
    message(STATUS "% Libreria movil NO compilada (BUILD_MOBILE_LIB=OFF)")
    message(STATUS "% Para compilar: cmake -DBUILD_MOBILE_LIB=ON ..")
endif()

# RESUMEN DE BUILD
message(STATUS "")
message(STATUS "Proyecto: ${PROJECT_NAME}")
message(STATUS "Estandar C++: ${CMAKE_CXX_STANDARD}")
message(STATUS "Directorio fuente: ${CMAKE_SOURCE_DIR}")
message(STATUS "Targets compilados:")
foreach(target ${ALL_TARGETS})
    if(TARGET ${target})
        message(STATUS "  * ${target}")
    endif()
endforeach()
if(TARGET voz_mobile)
    message(STATUS "  * voz_mobile (shared library para FFI)")
endif()
message(STATUS "-> Targets configurados correctamente con prefijo 'voz_'")
🎙️ Sistema Biometrico Multimodal de Voz
Sistema de autenticación biometrica basado en características vocales con texto dinámico, desarrollado en C++ 
para máximo rendimiento y eficiencia en recursos.

Pipeline completo: Normalizacion → VAD → STFT → MFCC → SVM

📁 biometria_voz/
├──📁 data/							 # Almacenamiento de datos build para dockerizar
├──📁 voz/							 # Código fuente del backend biometrico
│   └─📁 apps/						 # Aplicaciones principales   
│	 │
│	 ├──servidor_biometrico.cpp		      # main HTTP server limpio con el puerto 8080
│	 ├──📁 controller/                  
│	 │   |──usuario_controller.cpp/h      # Lógica coordinadora de endpoints usuarios   
│	 │   └──frases_controller.cpp/h       # Lógica coordinadora de endpoints frases   
|	 |   
│	 ├──📁 service/ 
│	 │   ├──autenticacion_service.cpp/h    # Lógica de registrar nuevo user y autenticar              
│	 │   |──registrar_service.cpp/h        # entrenamiento del nuevo usuario 
│	 │   │──frases_service.cpp/h           # Lógica de frases
│	 │   |──listar_service.cpp/h           # Lógica de listar
│	 │   └──listar_service.cpp/h           # Lógica de frases
|	 |
│	 ├──📁 testeo/                       # Ejecutables para testeo   
│	 │   ├──asr_server.cpp               # Servidor HTTP ASR
│	 │   ├──test_asr.cpp                 # Pruebas reconocimiento ASR
│	 │   ├──pruebasUnitarias.cpp         # Test sistema completo
│	 │   ├──procesar_dataset.cpp         # Preprocesamiento + división 80/20
│	 │   |──entrenar_modelo.cpp          # Entrenamiento SVM One-vs-All
│	 │   |──exportar_audio/features.cpp
│	 │   └──verificar_modelo/verificar_dataset.cpp
|	 │
│	 ├──📁 core/                       # Núcleo del sistema
│	 │   ├──📁 load_audio/            
│    │	 │   ├─libs.cpp                # Implementaciones auxiliares
│	 │   │   └─audio_io.cpp/h          # Decodificador MP3/WAV/FLAC
│	 │   ├──📁 augmentation/                          
│	 │   │   └─audio_augmentation.cpp/h    # Aumento de datos de audio
│	 │   ├──📁 preprocessing/            
│	 │   │   ├─preprocesar.h               # API pública de preprocesamiento
│	 │   │   ├─normalization.cpp           # 
│	 │   │   └─vad.cpp
│	 │   ├──📁 segmentation/             
│	 │   │   └─stft.cpp/h                  # Short-Time Fourier Transform
│	 │   ├──📁 features/                
│	 │   │   └─mfcc.cpp/h                 # Coeficientes MFCC + estadísticas
│	 │   ├──📁 classification/      
│	 │   │   ├─svm.h                  # API pública única
│	 │   │   ├─svm_core.cpp           # Predicción, scoring
│	 │   │   ├─svm_train.cpp          # Entrenamiento One-vs-All
│	 │   │   ├─svm_metrics.cpp        # Metricas, matrices confusión
│	 │   │   ├─svm_io.cpp             # Load/save modelos + datasets
│	 │   │   └─svm_utils.cpp          # Normalización, diagnóstico
│	 │   ├──📁 process_dataset/            
│	 │   │   ├─dataset.h              # API pública
│	 │   │   ├─dataset_io.cpp         # Load/save datasets binarios + mapeos
│	 │   │   ├─dataset_split.cpp      # Split train/test estrategico
│	 │   │   └─dataset_utils.cpp      # Validación, mapeo, estadísticas, utils
│	 │   ├──📁 pipeline/            
│	 │   │   ├─audio_pipeline.h       # API pública
│	 │   │   └─audio_pipeline.cpp     # flujo completo: preproc, segmentación, extracción...
│	 │   └──📁 asr/                  # Reconocimiento de voz
│	 │      ├─whisper_asr.cpp/h      # Integración Whisper
│	 │      ├─similaridad.cpp/h      # Comparación textual
│	 │      ├─httplib.h              # Servidor HTTP header-only
│	 │      ├─📁 models/             # Modelos Whisper
│	 │      └─📁 whisper/            # Binarios Whisper necesarios
│	 │ 
│	 ├──📁 utils/                   # Herramientas auxiliares
│	 │   ├──config.h				 # Configuración global (rutas, parámetros, etc.)
│	 │   └──audio_export.cpp/h       # Exportación de audio/features a CSV
│	 │
│	 ├──📁 external/                # Librerías externas header-only
│	 │   ├──httplib.h               # Servidor HTTP
│	 │   ├──json.hpp	             # JSON para C++
│	 │   ├──dr_wav.h                # Decodificador WAV
│	 │   ├──dr_flac.h               # Decodificador FLAC
│	 │   ├──minimp3.h               # Decodificador MP3
│	 │   └──minimp3_ex.h            # Extensiones MP3
│	 │
│	 ├──CMakeLists.txt          # Configuración build
│	 ├──README.md               # Documentación del backend
│	 └──shema.sql				# Esquema base de la BD 
│
├── .gitignore                # Archivos ignorados por git
├── .dockerignore             # Archivos ignorados por Docker
├── CMakeLists.txt            # Configuración del proyecto en MVC
├── CMakePresets.json         # Configuración CMake
├── copy_build.sh             # Script para copiar build al contenedor
├── docker-compose.yml        # Containerización
└── Dockerfile                # Imagen Docker

# Documentación del proyecto-------------------------------------------------------------------------------

* Descripción del Proyecto----------------------------------------------------------------------------------
	Este proyecto implementa un Sistema Biometrico Multimodal Backend enfocado actualmente en la autenticación de voz con 
	texto dinámico, con el objetivo de garantizar alta seguridad, eficiencia en recursos.

⚙️ Lineamientos Generales de Desarrollo-------------------------------------------------------------------
	-Objetivo principal: construir un sistema ligero, rápido, paralelizable y fácil de mantener, basado exclusivamente en 
	tecnicas tradicionales (sin redes neuronales).

🧠 Arquitectura modular basada en etapas:------------------------------------------------------------------
	-Preprocesamiento de audio donde se trata el audio para reducir un porcentaje el ruido y silencio, resaltar la voz, 
	  ya que no se puede limpiar al 100% el audip.
	-Segmentación: análisis espectral mediante STFT.
	-Extracción de características: coeficientes MFCC y estadísticas.
	-Clasificación: modelo SVM One-vs-All entrenado y mas.

🧩 Requisitos de Implementación----------------------------------------------------------------------------
	Todo el backend está desarrollado en C/C++, utilizando CMake y pensado para su dockerización.
	Se debe garantizar el uso de procesamiento paralelo (OpenMP) en todas las etapas críticas si son posibless.
	Evitando clases genericas, herencias innecesarias o arquitecturas sobreingenierizadas.
	Debemos mantener una lógica explícita y clara, usando estructuras simples y eficientes.
	
💡 Estilo y Convenciones del Código-----------------------------------------------------------------------
	No usar decoraciones como emojis en la salida por consola. Usar caracteres como ->, *, #, @, %, etc.
	Las salidas de consola deben ser informativas, claras y útiles para debug y producción, sin tildes.
	El código debe ser portable, escalable y fáciles de mantener.
	Siempre mantener la compatibilidad entre los diferentes modulos y siempre poner logs para debuggear.
	
🔌 Endpoints Disponibles---------------------------------------------------------------------------------
	-Registrar biometria de usuario existente: POST /voz/registrar_biometria
		POST http://localhost:8081/voz/registrar_biometria
		Body → form-data
		┌────────────────┬──────────┬─────────────────────┐
		│ ☑              │ Key      │ Value               │
		├────────────────┼──────────┼─────────────────────┤
		│ ✅             │identificador│ [Text] 1234567890│
		│ ✅             │ audios   │ [File]   audio1.flac│
		│ ✅             │ audios   │ [File]   audio2.flac│
		│ ✅             │ audios   │ [File]   audio3.flac│
		│ ✅             │ audios   │ [File]   audio4.flac│
		│ ✅             │ audios   │ [File]   audio5.flac│
		│ ✅             │ audios   │ [File]   audio6.flac│
		└────────────────┴──────────┴─────────────────────┘

	-Autenticar usuario: POST /voz/autenticar
		POST http://localhost:8081/voz/autenticar
		REQUIERE los 3 campos obligatorios: audio, identificador y id_frase
		Body → form-data
		┌────────────────┬──────────────┬────────────────────────┐
		│ ☑              │ Key          │ Value                  │
		├────────────────┼──────────────┼────────────────────────┤
		│ ✅             │ audio        │ [File] audio_auth.flac │
		│ ✅             │ identificador│ [Text] 1254444448      │
		│ ✅             │ id_frase     │ [Text] 1               │
		└────────────────┴──────────────┴────────────────────────┘
		
		Validaciones realizadas:
		* Autenticacion biometrica por SVM (audio)
		* Verificacion de frase dinamica (id_frase)
		* Validacion de identidad (identificador debe coincidir con resultado SVM)


	-Listar usuarios: GET /voz/usuarios
		GET http://localhost:8081/voz/usuarios

	-Eliminar usuario: DELETE /voz/usuarios/:id
		DELETE http://localhost:8081/voz/usuarios/1
		Elimina al usuario y su modelo biometrico (class_X.bin)
		Actualiza metadata.json automaticamente
		┌────────┬──────────┬─────────────────────┐
		│ Param  │ Type     │ Example             │
		├────────┼──────────┼─────────────────────┤
		│ id     │ Path     │ 1                   │
		└────────┴──────────┴─────────────────────┘

	-Desactivar/Activar credencial biometrica: PATCH /voz/credenciales/:id/estado
		PATCH http://localhost:8081/voz/credenciales/5/estado
		Actualiza el estado de una credencial biometrica en la BD
		Body → raw JSON
		┌─────────────────────────────────────────┐
		│ Content-Type: application/json          │
		├─────────────────────────────────────────┤
		│ {                                       │
		│   "estado": "inactivo"  // o "activo"   │
		│ }                                       │
		└─────────────────────────────────────────┘
		┌────────┬──────────┬─────────────────────┐
		│ Param  │ Type     │ Example             │
		├────────┼──────────┼─────────────────────┤
		│ id     │ Path     │ 5                   │
		└────────┴──────────┴─────────────────────┘

	-Listar frases disponibles: GET /listar/frases
		GET http://localhost:8081/listar/frases

	-Obtener frase especifica: GET /listar/frases?id=N
		GET http://localhost:8081/listar/frases?id=5
		┌────────┬──────────┬─────────────────────┐
		│ Param  │ Type     │ Example             │
		├────────┼──────────┼─────────────────────┤
		│ id     │ Query    │ 5                   │
		└────────┴──────────┴─────────────────────┘

	-Obtener frase aleatoria activa: GET /frases/aleatoria
		GET http://localhost:8081/frases/aleatoria
		Retorna una frase aleatoria marcada como activa

	-Agregar nueva frase: POST /agregar/frases
		POST http://localhost:8081/agregar/frases
		Body → raw JSON
		┌────────────────────────────────────────┐
		│ Content-Type: application/json         │
		├────────────────────────────────────────┤
		│ {                                      │
		│   "frase": "Mi voz es unica"           │
		│ }                                      │
		└────────────────────────────────────────┘

	-Activar/Desactivar frase: PATCH /frases/:id/estado
		PATCH http://localhost:8081/frases/5/estado
		Body → raw JSON
		┌─────────────────────────────────────────┐
		│ Content-Type: application/json          │
		├─────────────────────────────────────────┤
		│ {                                       │
		│   "activo": 1    // 1=activo, 0=inactivo│
		│ }                                       │
		└─────────────────────────────────────────┘

	-Eliminar frase: DELETE /frases/:id
		DELETE http://localhost:8081/frases/5
		┌────────┬──────────┬─────────────────────┐
		│ Param  │ Type     │ Example             │
		├────────┼──────────┼─────────────────────┤
		│ id     │ Path     │ 5                   │
		└────────┴──────────┴─────────────────────┘

🚀 Instrucciones para Despliegue con Docker----------------------------------------------------------------
	# 1. Copiar el build
	./copy_builds.ps1
	# 2. Limpiar
	docker-compose down --volumes
	# 3. Reconstruir
	docker-compose build --no-cache
	# 4. Levantar
	docker-compose up -d
	# 5. Reconstruir si hay cambios
	docker-compose up --build -d
	# bash# Conectar con el usuario desde Docker Desktop (Tab Exec)
	psql -U biometria -d usuarios_db
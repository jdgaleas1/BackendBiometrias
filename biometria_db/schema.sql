-- 📌 Tabla principal de usuarios
CREATE TABLE usuarios (
    id_usuario SERIAL PRIMARY KEY,
    nombres VARCHAR(100),
    apellidos VARCHAR(100),
    fecha_nacimiento DATE,
    sexo VARCHAR(10),
    identificador_unico VARCHAR(100) UNIQUE NOT NULL, -- correo o cedula
    estado VARCHAR(20) DEFAULT 'activo',
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 📌 Tabla para credenciales biometricas (oreja y voz)
CREATE TABLE credenciales_biometricas (
    id_credencial SERIAL PRIMARY KEY,
    id_usuario INTEGER REFERENCES usuarios(id_usuario) ON DELETE CASCADE,
    tipo_biometria VARCHAR(20) CHECK (tipo_biometria IN ('oreja', 'voz')),
    fecha_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    estado VARCHAR(20) DEFAULT 'activo',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 📌 Tabla de frases dinámicas (solo para voz)
CREATE TABLE textos_dinamicos_audio (
    id_texto SERIAL PRIMARY KEY,
    frase TEXT NOT NULL,
    estado_texto VARCHAR(20) DEFAULT 'activo', -- 'activo', 'usado', 'expirado'
    contador_usos INT DEFAULT 0,
    limite_usos INT DEFAULT 150,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 📌 Tabla de auditoría de validaciones biometricas
CREATE TABLE validaciones_biometricas (
    id_validacion SERIAL PRIMARY KEY,
    id_usuario INTEGER REFERENCES usuarios(id_usuario) ON DELETE SET NULL,
    id_credencial INTEGER,
    tipo_biometria VARCHAR(20),
    ip_cliente VARCHAR(45),
    ubicacion VARCHAR(200),
    dispositivo VARCHAR(200),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);  

-- 📌 Tabla de caracteristicas de hablantes (vectores MFCC)
CREATE TABLE caracteristicas_hablantes (
    id_caracteristica SERIAL PRIMARY KEY,
    id_usuario INTEGER REFERENCES usuarios(id_usuario) ON DELETE CASCADE,
    id_credencial INTEGER REFERENCES credenciales_biometricas(id_credencial),
    vector_features BYTEA NOT NULL,
    dimension INTEGER NOT NULL,
    origen VARCHAR(20) DEFAULT 'mobile',
    uuid_dispositivo VARCHAR(100),
    fecha_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_caracteristicas_usuario ON caracteristicas_hablantes(id_usuario);

-- 📌 Tabla de caracteristicas de oreja (vectores LBP)
CREATE TABLE caracteristicas_oreja (
    id_caracteristica SERIAL PRIMARY KEY,
    id_usuario INTEGER REFERENCES usuarios(id_usuario) ON DELETE CASCADE,
    id_credencial INTEGER REFERENCES credenciales_biometricas(id_credencial),
    vector_features BYTEA NOT NULL,
    dimension INTEGER NOT NULL,
    origen VARCHAR(20) DEFAULT 'mobile',
    uuid_dispositivo VARCHAR(100),
    fecha_captura TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_caracteristicas_oreja_usuario ON caracteristicas_oreja(id_usuario);

-- Crear el rol que PostgREST usará
CREATE ROLE web_anon NOLOGIN;

-- Dar acceso al esquema público
GRANT USAGE ON SCHEMA public TO web_anon;

-- Dar acceso total a las tablas
GRANT ALL ON usuarios TO web_anon;
GRANT ALL ON credenciales_biometricas TO web_anon;
GRANT ALL ON textos_dinamicos_audio TO web_anon;
GRANT ALL ON validaciones_biometricas TO web_anon;
GRANT ALL ON caracteristicas_hablantes TO web_anon;
GRANT ALL ON caracteristicas_oreja TO web_anon;

-- Dar permisos a las secuencias (NOMBRES CORRECTOS)
GRANT ALL ON SEQUENCE usuarios_id_usuario_seq TO web_anon;
GRANT ALL ON SEQUENCE credenciales_biometricas_id_credencial_seq TO web_anon;
GRANT ALL ON SEQUENCE textos_dinamicos_audio_id_texto_seq TO web_anon;
GRANT ALL ON SEQUENCE validaciones_biometricas_id_validacion_seq TO web_anon;
GRANT ALL ON SEQUENCE caracteristicas_hablantes_id_caracteristica_seq TO web_anon;
GRANT ALL ON SEQUENCE caracteristicas_oreja_id_caracteristica_seq TO web_anon;

-- Indices para sincronizacion
CREATE INDEX idx_usuarios_updated ON usuarios(updated_at);
CREATE INDEX idx_credenciales_updated ON credenciales_biometricas(updated_at);
CREATE INDEX idx_textos_updated ON textos_dinamicos_audio(updated_at);

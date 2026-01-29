# ============================================================================
# Script: Compilar libvoz_mobile.so y Preparar Entrega para Flutter
# Proyecto: Sistema Biometrico Multimodal - Modulo Movil
# Fecha: 19 de enero de 2026
# ============================================================================

param(
    [string]$AndroidABI = "arm64-v8a",  # arm64-v8a, armeabi-v7a, x86_64, x86
    [string]$AndroidPlatform = "android-24",
    [string]$OutputDir = "D:\entrega_flutter_mobile"
)

$rootDir = "D:\server\biometria_voz"

# ============================================================================
# FASE 1: COMPILAR LIBRERIA MOVIL
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  COMPILAR LIBRERIA MOVIL (ANDROID)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Buscar Android NDK (busqueda exhaustiva en multiples rutas)
Write-Host "-> Buscando Android NDK..." -ForegroundColor Yellow

# Construir lista de rutas potenciales
$sdkPaths = @()

# Buscar en todos los perfiles de usuario (C:\Users\*)
Get-ChildItem "C:\Users" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $sdkPaths += Join-Path $_.FullName "AppData\Local\Android\Sdk"
}

# Agregar otras ubicaciones comunes
$sdkPaths += "C:\Android\Sdk"
$sdkPaths += "D:\Android\Sdk"

$ndkFound = $null

foreach ($sdkPath in $sdkPaths) {
    if (-not (Test-Path $sdkPath)) { continue }

    Write-Host "   @ Intentando SDK: $sdkPath" -ForegroundColor DarkGray

    # Intentar ndk-bundle (formato antiguo)
    $ndkBundle = Join-Path $sdkPath "ndk-bundle"
    if ((Test-Path $ndkBundle) -and (Test-Path (Join-Path $ndkBundle "build\cmake\android.toolchain.cmake"))) {
        $ndkFound = $ndkBundle
        Write-Host "   @ NDK encontrado (ndk-bundle): $ndkFound" -ForegroundColor Green
        break
    }

    # Intentar ndk/<version> (formato moderno - side by side)
    $ndkDir = Join-Path $sdkPath "ndk"
    if (Test-Path $ndkDir) {
        $versions = Get-ChildItem $ndkDir -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        foreach ($version in $versions) {
            $toolchainPath = Join-Path $version.FullName "build\cmake\android.toolchain.cmake"
            if (Test-Path $toolchainPath) {
                $ndkFound = $version.FullName
                Write-Host "   @ NDK encontrado (version $($version.Name)): $ndkFound" -ForegroundColor Green
                break
            }
        }
        if ($ndkFound) { break }
    }
}

if (-not $ndkFound) {
    Write-Host "`n# ERROR: Android NDK no encontrado" -ForegroundColor Red
    Write-Host "# Instala NDK desde Android Studio -> SDK Manager -> NDK (Side by side)" -ForegroundColor Yellow
    Write-Host "# O establece la variable ANDROID_NDK manualmente:" -ForegroundColor Yellow
    Write-Host '   $env:ANDROID_NDK = "C:\Users\tu_usuario\AppData\Local\Android\Sdk\ndk\XX.X.XXXX"' -ForegroundColor DarkGray
    exit 1
}

Write-Host "-> Android NDK encontrado: $ndkFound" -ForegroundColor Green
$env:ANDROID_NDK = $ndkFound

# Configurar directorios
$buildDir = "$rootDir\build-mobile-$AndroidABI"

Write-Host "-> ABI: $AndroidABI" -ForegroundColor White
Write-Host "-> Platform: $AndroidPlatform" -ForegroundColor White
Write-Host "-> Build dir: $buildDir`n" -ForegroundColor White

# Limpiar build anterior
if (Test-Path $buildDir) {
    Write-Host "-> Limpiando build anterior..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}

New-Item -ItemType Directory -Path $buildDir | Out-Null

# Configurar CMake para Android
Write-Host "`n[1/2] Configurando CMake para Android...`n" -ForegroundColor Green

cd $buildDir

$cmakeArgs = @(
    "..",
    "-G", "Ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$ndkFound/build/cmake/android.toolchain.cmake",
    "-DANDROID_ABI=$AndroidABI",
    "-DANDROID_PLATFORM=$AndroidPlatform",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_MOBILE_LIB=ON"
)

& cmake @cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n# ERROR: Fallo configuracion de CMake" -ForegroundColor Red
    exit 1
}

# Compilar solo voz_mobile
Write-Host "`n[2/2] Compilando libvoz_mobile.so...`n" -ForegroundColor Green

cmake --build . --target voz_mobile --parallel 8

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n# ERROR: Fallo compilacion de libvoz_mobile.so" -ForegroundColor Red
    exit 1
}

# Verificar resultado
$libPath = "$buildDir\voz\libvoz_mobile.so"
if (Test-Path $libPath) {
    $size = (Get-Item $libPath).Length / 1MB
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  COMPILACION EXITOSA" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    Write-Host "Libreria: $libPath" -ForegroundColor Green
    Write-Host "Tamano: $([math]::Round($size, 2)) MB" -ForegroundColor Green
    Write-Host "ABI: $AndroidABI`n" -ForegroundColor Green
} else {
    Write-Host "`n# ERROR: libvoz_mobile.so no encontrada" -ForegroundColor Red
    exit 1
}

cd $rootDir

# ============================================================================
# FASE 2: PREPARAR ENTREGA PARA FLUTTER
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  PREPARANDO ENTREGA PARA FLUTTER" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$buildDirVoz = "$buildDir\voz"

# Crear directorio de salida
Write-Host "-> Creando directorio de entrega: $OutputDir" -ForegroundColor Yellow
if (Test-Path $OutputDir) {
    Remove-Item -Recurse -Force $OutputDir
}
New-Item -ItemType Directory -Path $OutputDir | Out-Null

# ============================================================================
# 1. COPIAR LIBRERIAS COMPILADAS
# ============================================================================

Write-Host "`n[1/4] Copiando librerias compiladas..." -ForegroundColor Green

$libDir = "$OutputDir\libraries\android\arm64-v8a"
New-Item -ItemType Directory -Path $libDir -Force | Out-Null

if (Test-Path "$buildDirVoz\libvoz_mobile.so") {
    Copy-Item "$buildDirVoz\libvoz_mobile.so" -Destination $libDir
    $size = (Get-Item "$libDir\libvoz_mobile.so").Length / 1MB
    Write-Host "   @ libvoz_mobile.so copiada [$([math]::Round($size, 2)) MB]" -ForegroundColor White
} else {
    Write-Host "   # ERROR: libvoz_mobile.so no encontrada en:" -ForegroundColor Red
    Write-Host "      $buildDirVoz\libvoz_mobile.so" -ForegroundColor Red
    exit 1
}

if (Test-Path "$buildDirVoz\libsqlite3_local.a") {
    Copy-Item "$buildDirVoz\libsqlite3_local.a" -Destination $libDir
    $size = (Get-Item "$libDir\libsqlite3_local.a").Length / 1MB
    Write-Host "   @ libsqlite3_local.a copiada [$([math]::Round($size, 2)) MB]" -ForegroundColor White
}

# ============================================================================
# 2. COPIAR MODELOS SVM
# ============================================================================

Write-Host "`n[2/4] Copiando modelos SVM..." -ForegroundColor Green

$modelsDir = "$OutputDir\assets\models\v1"
New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

if (Test-Path "$rootDir\models\v1") {
    Copy-Item "$rootDir\models\v1\*" -Destination $modelsDir -Recurse -Force

    $classFiles = (Get-ChildItem "$modelsDir\class_*.bin").Count
    $totalSize = (Get-ChildItem "$modelsDir\*" -File | Measure-Object -Property Length -Sum).Sum / 1MB

    Write-Host "   @ $classFiles archivos class_*.bin copiados" -ForegroundColor White
    Write-Host "   @ metadata.json copiado" -ForegroundColor White
    Write-Host "   @ Tamano total: $([math]::Round($totalSize, 2)) MB" -ForegroundColor White
} else {
    Write-Host "   # ERROR: Directorio models/v1 no encontrado" -ForegroundColor Red
    exit 1
}

# ============================================================================
# 3. COPIAR CARACTERISTICAS (DATASETS)
# ============================================================================

Write-Host "`n[3/4] Copiando datasets de caracteristicas..." -ForegroundColor Green

$featuresDir = "$OutputDir\assets\caracteristicas\v1"
New-Item -ItemType Directory -Path $featuresDir -Force | Out-Null

if (Test-Path "$rootDir\caracteristicas\v1") {
    Copy-Item "$rootDir\caracteristicas\v1\*.dat" -Destination $featuresDir -Force

    if (Test-Path "$featuresDir\caracteristicas_train.dat") {
        $trainSize = (Get-Item "$featuresDir\caracteristicas_train.dat").Length / 1MB
        Write-Host "   @ caracteristicas_train.dat copiado [$([math]::Round($trainSize, 2)) MB]" -ForegroundColor White
    }

    if (Test-Path "$featuresDir\caracteristicas_test.dat") {
        $testSize = (Get-Item "$featuresDir\caracteristicas_test.dat").Length / 1MB
        Write-Host "   @ caracteristicas_test.dat copiado [$([math]::Round($testSize, 2)) MB]" -ForegroundColor White
    }
} else {
    Write-Host "   # ERROR: Directorio caracteristicas/v1 no encontrado" -ForegroundColor Red
    exit 1
}

# ============================================================================
# 4. COPIAR HEADERS Y APIS
# ============================================================================

Write-Host "`n[4/4] Copiando headers y apis..." -ForegroundColor Green

$docsDir = "$OutputDir\apis"
New-Item -ItemType Directory -Path $docsDir -Force | Out-Null

Copy-Item "$rootDir\voz\apps\mobile\mobile_api.h" -Destination "$docsDir\mobile_api.h"
Write-Host "   @ mobile_api.h copiado" -ForegroundColor White

Copy-Item "$rootDir\voz\apps\mobile\sqlite_adapter.h" -Destination "$docsDir\sqlite_adapter.h"
Write-Host "   @ sqlite_adapter.h copiado" -ForegroundColor White

# ============================================================================
# RESUMEN FINAL
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ENTREGA PREPARADA EXITOSAMENTE" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Ubicacion: $OutputDir`n" -ForegroundColor Yellow

# Calcular tamano total
$totalSize = (Get-ChildItem -Path $OutputDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "Tamano total: $([math]::Round($totalSize, 2)) MB`n" -ForegroundColor White

Write-Host "Archivos incluidos:" -ForegroundColor White
Write-Host "  @ 1 libreria compilada (.so)" -ForegroundColor Gray
Write-Host "  @ 68 archivos de modelo SVM (.bin)" -ForegroundColor Gray
Write-Host "  @ 2 datasets de caracteristicas (.dat)" -ForegroundColor Gray
Write-Host "  @ 2 archivos de apis (.h)" -ForegroundColor Gray

Write-Host "`nProximo paso:" -ForegroundColor Yellow
Write-Host "  -> Entregar carpeta '$OutputDir' al equipo Flutter" -ForegroundColor White
Write-Host "  -> Indicar que lean 'ENTREGA_EQUIPO_FLUTTER.md' primero`n" -ForegroundColor White

# Abrir directorio
Write-Host "Desea abrir el directorio de entrega? (S/N): " -NoNewline -ForegroundColor Yellow
$respuesta = Read-Host
if ($respuesta -eq "S" -or $respuesta -eq "s") {
    Start-Process explorer.exe -ArgumentList $OutputDir
}

Write-Host "`n@ Proceso completado exitosamente`n" -ForegroundColor Green

# Downloads the official person_detect int8 model as a C++ translation unit (same as ESP-IDF example).
# Run from repo root or here:  powershell -ExecutionPolicy Bypass -File scripts/fetch_person_detect_model.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$uri = 'https://raw.githubusercontent.com/espressif/esp-tflite-micro/master/examples/person_detection/main/person_detect_model_data.cc'
$dest = Join-Path $root 'src\person_detect_model_data.cpp'
Write-Host "Downloading model (~1.9 MB) ..."
Invoke-WebRequest -Uri $uri -OutFile $dest -UseBasicParsing
Write-Host "Wrote $dest"

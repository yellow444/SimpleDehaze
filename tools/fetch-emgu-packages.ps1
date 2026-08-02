#requires -Version 7.0
<#
.SYNOPSIS
    Скачивает три 'особых' GPU-пакета Emgu CV в локальный фид localpackages\,
    чтобы проект восстанавливался и работал на GPU, не храня гигабайты в git.

.DESCRIPTION
    Эти три .nupkg убраны с nuget.org (cuDNN-DLL внутри весит 688 МБ, а лимит пакета
    на nuget.org - 250 МБ), но официально выложены ассетами GitHub-релиза Emgu:
      https://github.com/emgucv/emgucv/releases/tag/4.8.0
    Скрипт качает их в localpackages\ (папка в .gitignore). Остальные CUDA-библиотеки
    (редистрибутивы NVIDIA) тянутся обычным образом с nuget.org при `dotnet restore`.

    После запуска:  dotnet restore  &&  dotnet build -c Release

.EXAMPLE
    pwsh tools\fetch-emgu-packages.ps1
    pwsh tools\fetch-emgu-packages.ps1 -Force      # перекачать заново
#>
[CmdletBinding()]
param(
    [string]$Tag = '4.8.0',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$feed     = Join-Path $repoRoot 'localpackages'
New-Item -ItemType Directory -Force -Path $feed | Out-Null

$base = "https://github.com/emgucv/emgucv/releases/download/$Tag"
$packages = @(
    'Emgu.CV.runtime.windows.cuda.4.8.0.5324.nupkg',     # OpenCV+CUDA (opencv_world)
    'Emgu.runtime.windows.cuda.blas.lt.12.0.104.nupkg',  # cuBLASLt 12.0 (нет на nuget.org)
    'Emgu.runtime.windows.cuda.dnn.cnn.infer.8.8.0.nupkg' # cuDNN cnn-infer 8.8 (нет на nuget.org)
)

$ProgressPreference = 'SilentlyContinue'   # иначе прогресс-бар IWR сильно тормозит крупные загрузки
$downloaded = 0
foreach ($p in $packages) {
    $dest = Join-Path $feed $p
    if ((Test-Path $dest) -and -not $Force -and (Get-Item $dest).Length -gt 50MB) {
        Write-Host ("уже есть: {0}" -f $p) -ForegroundColor DarkGray
        continue
    }
    $url = "$base/$p"
    Write-Host ("качаю: {0}" -f $p) -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $dest -MaximumRedirection 10
    $mb = (Get-Item $dest).Length / 1MB
    if ($mb -lt 50) { throw "Файл '$p' подозрительно мал ($([int]$mb) МБ) - проверь -Tag/доступ к GitHub." }
    Write-Host ("  ok: {0:N1} МБ" -f $mb) -ForegroundColor Green
    $downloaded++
}

$sum = (Get-ChildItem $feed -Filter *.nupkg | Measure-Object Length -Sum).Sum / 1MB
Write-Host ""
Write-Host ("Локальный фид готов: {0}  ({1:N0} МБ, скачано {2})" -f $feed, $sum, $downloaded) -ForegroundColor Green
Write-Host "Дальше:  dotnet restore .\SimpleDeHaze\SimpleDeHaze.csproj  &&  dotnet build -c Release"

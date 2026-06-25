#requires -Version 7.0
<#
    Сборка одной командой: скачать GPU-зависимости Emgu (если их ещё нет)
    и собрать проект.

    Примеры:
        pwsh build.ps1              # Release
        pwsh build.ps1 -Configuration Debug
#>
param([string]$Configuration = 'Release')

$ErrorActionPreference = 'Stop'

# 1) локальный фид GPU-пакетов (качается из GitHub-релиза Emgu, см. tools\)
& "$PSScriptRoot\tools\fetch-emgu-packages.ps1"

# 2) сборка
dotnet build "$PSScriptRoot\SimpleDeHaze\SimpleDeHaze.csproj" -c $Configuration

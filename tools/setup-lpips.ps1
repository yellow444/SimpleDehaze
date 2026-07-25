param(
    [double]$MinimumFreeReserveGiB = 10
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$benchdata = Join-Path $repo 'benchdata'
$packages = Join-Path $benchdata 'python_packages'
$modelCache = Join-Path $benchdata 'model_cache'
$checkpoint = Join-Path $modelCache 'hub\checkpoints\alexnet-owt-7be5be79.pth'
$expectedCheckpointBytes = 244408911L
$expectedCheckpointSha256 = '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02'

$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($benchdata))
$required = [long]($MinimumFreeReserveGiB * 1GB) + 300MB
Write-Host ("SPACE LPIPS: free={0:N2} GiB required={1:N2} GiB reserve={2:N2} GiB" -f ($drive.AvailableFreeSpace/1GB), ($required/1GB), $MinimumFreeReserveGiB)
if ($drive.AvailableFreeSpace -lt $required) { throw 'Not enough disk space for LPIPS package/checkpoint plus reserve.' }

New-Item -ItemType Directory -Force -Path $packages,$modelCache | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$packages$([System.IO.Path]::PathSeparator)$env:PYTHONPATH" } else { $packages }
$env:TORCH_HOME = $modelCache

& python -c "import torch, torchvision; print('PYTORCH', torch.__version__, 'TORCHVISION', torchvision.__version__, 'CUDA', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'A working PyTorch + torchvision installation is required for LPIPS.' }

if (-not (Test-Path -LiteralPath (Join-Path $packages 'lpips\__init__.py') -PathType Leaf)) {
    & python -m pip install --target $packages --no-deps lpips==0.1.4
    if ($LASTEXITCODE -ne 0) { throw 'LPIPS package installation failed.' }
}

if (Test-Path -LiteralPath $checkpoint -PathType Leaf) {
    $info = Get-Item -LiteralPath $checkpoint
    $hash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($info.Length -ne $expectedCheckpointBytes -or $hash -ne $expectedCheckpointSha256) {
        throw "Existing AlexNet checkpoint failed verification: bytes=$($info.Length), sha256=$hash"
    }
} else {
    & python -c "import lpips; model=lpips.LPIPS(net='alex', version='0.1'); model.eval(); print('LPIPS-CHECKPOINT-DOWNLOADED')"
    if ($LASTEXITCODE -ne 0) { throw 'LPIPS/AlexNet initialization failed.' }
    $info = Get-Item -LiteralPath $checkpoint
    $hash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($info.Length -ne $expectedCheckpointBytes -or $hash -ne $expectedCheckpointSha256) {
        throw "Downloaded AlexNet checkpoint failed verification: bytes=$($info.Length), sha256=$hash"
    }
}

$versionsJson = & python -c "import json,torch,torchvision,lpips; print(json.dumps({'python':__import__('sys').version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__,'lpips':'0.1.4','cuda_available':torch.cuda.is_available()}))"
if ($LASTEXITCODE -ne 0) { throw 'Could not capture LPIPS environment versions.' }
$versions = $versionsJson | Select-Object -Last 1 | ConvertFrom-Json
$record = [ordered]@{
    generatedUtc = [DateTime]::UtcNow.ToString('o')
    versions = $versions
    model = [ordered]@{
        network = 'alex'
        lpipsVersion = 'v0.1'
        checkpoint = [System.IO.Path]::GetRelativePath($benchdata, $checkpoint).Replace('\','/')
        bytes = $expectedCheckpointBytes
        sha256 = $expectedCheckpointSha256
    }
}
$recordPath = Join-Path $modelCache 'simpledehaze-lpips-environment.json'
$record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $recordPath -Encoding utf8
Write-Host "LPIPS-READY $recordPath free_after=$([math]::Round($drive.AvailableFreeSpace/1GB,2)) GiB"

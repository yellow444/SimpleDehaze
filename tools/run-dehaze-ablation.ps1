param(
    [string]$Manifest = (Join-Path $PSScriptRoot '..\datasets\manifests\o-haze-in-repo.json'),
    [string]$Output = (Join-Path $PSScriptRoot '..\benchmark_results\ablation'),
    [ValidateSet('val', 'test', 'all')]
    [string]$Split = 'test',
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$project = Join-Path $repo 'SimpleDeHaze\SimpleDeHaze.csproj'
$manifestPath = [System.IO.Path]::GetFullPath($Manifest)
$outputRoot = [System.IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$common = @(
    '--benchmark', "--manifest=$manifestPath", "--split=$Split", '--profile=core', '--evalfull'
)
if ($Quick) {
    $common += @('--limit=3', '--maxdim=512', '--warmup=0', '--repeat=1', '--no-memory')
} else {
    $common += @('--native', '--warmup=1', '--repeat=3')
}

$cases = @(
    @{ Name='canonical_srgb'; Method='DCP канонический'; Extra=@() },
    @{ Name='canonical_linear'; Method='DCP канонический'; Extra=@('--linear') },
    @{ Name='rfep_standard_linear'; Method='Boundary-Constrained'; Extra=@('--linear', '--params=chroma=0,csbound=0,strict=1,bscale=1,bmax=1,rho=1') },
    @{ Name='rfep_chromasafe_linear'; Method='Boundary-Constrained'; Extra=@('--linear', '--params=csbound=1,strict=1,bscale=1,bmax=1,rho=1') },
    @{ Name='rfep_chromasafe_relaxed'; Method='Boundary-Constrained'; Extra=@('--linear', '--params=csbound=1,strict=0') },
    @{ Name='roughness_two_scale'; Method='HSV +'; Extra=@('--params=rough=0') },
    @{ Name='roughness_five_scale_r2'; Method='HSV +'; Extra=@('--params=rough=1') },
    @{ Name='laplacian_smoothstep'; Method='Transmission-aware'; Extra=@('--params=wiener=0,rough=1') },
    @{ Name='laplacian_wiener'; Method='Transmission-aware'; Extra=@('--params=wiener=1,rough=1') }
)

$index = @()
foreach ($case in $cases) {
    $csv = Join-Path $outputRoot ($case.Name + '.csv')
    $arguments = @('run', '--project', $project, '-c', 'Release', '--no-build', '--') +
        $common + @("--methods=$($case.Method)", "--out=$csv") + $case.Extra
    Write-Host "[$($case.Name)]"
    & dotnet @arguments
    if ($LASTEXITCODE -ne 0) { throw "Ablation case failed: $($case.Name)" }
    $index += [ordered]@{
        name = $case.Name
        csv = [System.IO.Path]::GetRelativePath($outputRoot, $csv).Replace('\', '/')
        metadata = [System.IO.Path]::GetRelativePath($outputRoot, [System.IO.Path]::ChangeExtension($csv, '.meta.json')).Replace('\', '/')
        method_filter = $case.Method
        extra_arguments = $case.Extra
    }
}

$indexPath = Join-Path $outputRoot 'ablation-index.json'
[ordered]@{
    generated = (Get-Date).ToString('o')
    manifest = $manifestPath
    split = $Split
    quick = [bool]$Quick
    cases = $index
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $indexPath -Encoding utf8
Write-Host "Ablation index: $indexPath"

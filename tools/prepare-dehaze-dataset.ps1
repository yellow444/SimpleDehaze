param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('O-HAZE', 'I-HAZE', 'Dense-Haze', 'NH-HAZE')]
    [string]$Name,

    [string]$Source,
    [string]$Destination = (Join-Path $PSScriptRoot '..\benchdata'),
    [switch]$Download
)

$ErrorActionPreference = 'Stop'

$catalog = @{
    'O-HAZE' = @{
        Url = 'https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip'
        AcceptedPairCounts = @(45)
        PublishedPairs = 45
        Version = '2018'
        Homepage = 'https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/'
    }
    'I-HAZE' = @{
        Url = 'https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/I-HAZE.zip'
        # The dataset page describes 35 scenes, while the currently published
        # NTIRE archive contains 30 paired files (01-25 and 31-35).
        AcceptedPairCounts = @(30, 35)
        PublishedPairs = 35
        Version = '2018-official-archive'
        Homepage = 'https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/'
    }
    'Dense-Haze' = @{
        Url = 'https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/files/Dense_Haze_NTIRE19.zip'
        # The official page describes 33 pairs; the archive currently served by
        # its download link contains 55 numbered GT/hazy pairs.
        AcceptedPairCounts = @(33, 55)
        PublishedPairs = 33
        Version = '2019-official-archive'
        Homepage = 'https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/'
    }
    'NH-HAZE' = @{
        Url = 'https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/files/NH-HAZE.zip'
        AcceptedPairCounts = @(55)
        PublishedPairs = 55
        Version = '2020'
        Homepage = 'https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/'
    }
}

$definition = $catalog[$Name]
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$datasetRoot = Join-Path $destinationRoot $Name

if ($Download) {
    if ($Source) { throw 'Use either -Source or -Download, not both.' }
    New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
    $archive = Join-Path $destinationRoot ($Name + '.zip')
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Host "Downloading $($definition.Url)"
        $partialArchive = $archive + '.partial'
        Invoke-WebRequest -Uri $definition.Url -OutFile $partialArchive
        Move-Item -LiteralPath $partialArchive -Destination $archive
    }
    $reuseExtracted = $false
    if (Test-Path -LiteralPath $datasetRoot) {
        $existing = Get-ChildItem -LiteralPath $datasetRoot -Force
        $reuseExtracted = $existing.Count -gt 0
    }
    if ($reuseExtracted) {
        Write-Host "Reusing extracted dataset: $datasetRoot"
    } else {
        New-Item -ItemType Directory -Force -Path $datasetRoot | Out-Null
        Expand-Archive -LiteralPath $archive -DestinationPath $datasetRoot
    }
    $Source = $datasetRoot
}

if (-not $Source) { throw 'Specify an extracted dataset with -Source or use -Download.' }
$sourceRoot = (Resolve-Path -LiteralPath $Source).Path
$extensions = @('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
$images = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File |
    Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() }

function Get-PairKey([System.IO.FileInfo]$file) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($file.Name).ToLowerInvariant()
    $stem = $stem -replace '(ground[ _-]*truth|haze[ _-]*free|hazy|haze|clear|gt)', ''
    $stem = $stem -replace '[^a-z0-9]+', ''
    if ($stem -match '^(\d+)$') { return ([int]$Matches[1]).ToString('D4') }
    if ($stem -match '(\d+)') { return ([int]$Matches[1]).ToString('D4') }
    return $stem
}

function Is-Clear([System.IO.FileInfo]$file) {
    $text = ($file.FullName.Substring($sourceRoot.Length) + ' ' + $file.Name).ToLowerInvariant()
    return $text -match '(^|[^a-z])(gt|clear|ground[ _-]*truth|haze[ _-]*free)([^a-z]|$)'
}

function Relative-ToSource([string]$path) {
    return [System.IO.Path]::GetRelativePath($sourceRoot, $path).Replace('\', '/')
}

$clearByKey = @{}
foreach ($file in ($images | Where-Object { Is-Clear $_ })) {
    $key = Get-PairKey $file
    if (-not $clearByKey.ContainsKey($key)) { $clearByKey[$key] = $file }
}

$hazy = $images | Where-Object { -not (Is-Clear $_) } | Sort-Object FullName
$pairs = @()
for ($index = 0; $index -lt $hazy.Count; $index++) {
    $file = $hazy[$index]
    $key = Get-PairKey $file
    if (-not $clearByKey.ContainsKey($key)) { continue }
    $clear = $clearByKey[$key]
    $pairs += [ordered]@{
        id = $key
        hazy = Relative-ToSource $file.FullName
        clear = Relative-ToSource $clear.FullName
        split = if (($index % 2) -eq 0) { 'val' } else { 'test' }
        sha256Hazy = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        sha256Clear = (Get-FileHash -LiteralPath $clear.FullName -Algorithm SHA256).Hash
    }
}

if ($definition.AcceptedPairCounts -notcontains $pairs.Count) {
    $accepted = $definition.AcceptedPairCounts -join ' or '
    throw "Found $($pairs.Count) pairs, expected $accepted. Inspect dataset naming before benchmarking."
}

$manifest = [ordered]@{
    name = $Name
    version = $definition.Version
    homepage = $definition.Homepage
    publishedPairs = $definition.PublishedPairs
    archivePairs = $pairs.Count
    root = '.'
    pairs = $pairs
}
$manifestPath = Join-Path $sourceRoot 'simpledehaze-manifest.json'
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "Manifest: $manifestPath"
Write-Host "Pairs: $($pairs.Count); val=$((@($pairs | Where-Object split -eq 'val')).Count); test=$((@($pairs | Where-Object split -eq 'test')).Count)"

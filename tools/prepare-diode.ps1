param(
    [string]$SourceRoot = (Join-Path $PSScriptRoot '..\benchdata\rgbd_sources\diode_val'),
    [string]$Output = (Join-Path $PSScriptRoot '..\benchdata\manifests\diode-val-500.json'),
    [ValidateRange(1, 100000)]
    [int]$MaxFrames = 500,
    [string]$Seed = 'simpledehaze-diode-v1'
)

$ErrorActionPreference = 'Stop'
$source = [System.IO.Path]::GetFullPath($SourceRoot)
$outputPath = [System.IO.Path]::GetFullPath($Output)
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "DIODE source directory is absent: $source. Run tools/prepare-research-data.ps1 -Name DIODE-Val -Extract first."
}

function Get-StableKey([string]$value) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes("$Seed`n$value")
    $hash = [System.Security.Cryptography.SHA256]::HashData($bytes)
    return [Convert]::ToHexString($hash).ToLowerInvariant()
}

function To-PortableRelative([string]$path) {
    return [System.IO.Path]::GetRelativePath($source, $path).Replace('\', '/')
}

$depthFiles = @(Get-ChildItem -LiteralPath $source -Recurse -File -Filter '*_depth.npy' |
    Where-Object { $_.Name -notlike '*_depth_mask.npy' })
if ($depthFiles.Count -eq 0) { throw "No *_depth.npy files found below $source" }

$candidates = foreach ($depth in $depthFiles) {
    $stem = $depth.FullName.Substring(0, $depth.FullName.Length - '_depth.npy'.Length)
    $image = @('.png', '.jpg', '.jpeg') | ForEach-Object { $stem + $_ } | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    $mask = $stem + '_depth_mask.npy'
    if (-not $image -or -not (Test-Path -LiteralPath $mask -PathType Leaf)) { continue }

    $relative = To-PortableRelative $image
    $segments = $relative.Split('/')
    $sceneIndex = -1
    for ($i = 0; $i -lt $segments.Length; $i++) {
        if ($segments[$i] -match '^scene[_-]') { $sceneIndex = $i; break }
    }
    $domain = if ($relative -match '(^|/)(indoor|indoors)(/|$)') { 'indoor' } elseif ($relative -match '(^|/)(outdoor|outdoors)(/|$)') { 'outdoor' } else { 'unknown' }
    $sceneGroup = if ($sceneIndex -ge 0) { "$domain/$($segments[$sceneIndex])" } else { "$domain/$([System.IO.Path]::GetDirectoryName($relative).Replace('\','/'))" }

    [pscustomobject]@{
        id = [System.IO.Path]::GetFileName($stem)
        domain = $domain
        sceneGroup = $sceneGroup
        image = $relative
        depth = To-PortableRelative $depth.FullName
        mask = To-PortableRelative $mask
        selectionKey = Get-StableKey $relative
    }
}
if (@($candidates).Count -eq 0) { throw 'DIODE files were found, but no complete image/depth/mask triples exist.' }

# Prefer a balanced indoor/outdoor sample; deterministically fill any unused quota.
$selected = [System.Collections.Generic.List[object]]::new()
$knownDomains = @('indoor', 'outdoor')
$perDomain = [math]::Floor($MaxFrames / $knownDomains.Count)
foreach ($domain in $knownDomains) {
    @($candidates | Where-Object domain -eq $domain | Sort-Object selectionKey | Select-Object -First $perDomain) |
        ForEach-Object { $selected.Add($_) }
}
$selectedPaths = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($item in $selected) { [void]$selectedPaths.Add($item.image) }
if ($selected.Count -lt $MaxFrames) {
    @($candidates | Where-Object { -not $selectedPaths.Contains($_.image) } | Sort-Object selectionKey | Select-Object -First ($MaxFrames - $selected.Count)) |
        ForEach-Object { $selected.Add($_); [void]$selectedPaths.Add($_.image) }
}

# Partition scene groups, not individual frames: no scene can leak across splits. DIODE val has
# only six top-level physical scenes with unequal frame counts, so optimize the frame proportions
# over all 3^6 group assignments instead of blindly slicing the group list.
$groups = @($selected | Group-Object sceneGroup | ForEach-Object {
    [pscustomobject]@{ name = $_.Name; count = $_.Count; key = Get-StableKey $_.Name }
} | Sort-Object key)
$splitNames = @('development', 'validation', 'internal_test')
$splitByGroup = @{}
if ($groups.Count -le 12) {
    $targets = @((0.60 * $selected.Count), (0.20 * $selected.Count), (0.20 * $selected.Count))
    $bestError = [double]::PositiveInfinity
    $bestDigits = $null
    $combinations = [int][math]::Pow(3, $groups.Count)
    for ($code = 0; $code -lt $combinations; $code++) {
        $value = $code
        $counts = @(0, 0, 0)
        $digits = [int[]]::new($groups.Count)
        for ($i = 0; $i -lt $groups.Count; $i++) {
            $digits[$i] = $value % 3; $value = [math]::Floor($value / 3)
            $counts[$digits[$i]] += $groups[$i].count
        }
        if ($counts[0] -eq 0 -or $counts[1] -eq 0 -or $counts[2] -eq 0) { continue }
        $splitScore = 0.0
        for ($s = 0; $s -lt 3; $s++) { $splitScore += [math]::Pow(($counts[$s] - $targets[$s]) / $selected.Count, 2) }
        if ($splitScore -lt $bestError) { $bestError = $splitScore; $bestDigits = $digits.Clone() }
    }
    if (-not $bestDigits) { throw 'Could not produce three non-empty scene-level splits.' }
    for ($i = 0; $i -lt $groups.Count; $i++) { $splitByGroup[$groups[$i].name] = $splitNames[$bestDigits[$i]] }
    $splitPolicy = 'exhaustive deterministic scene-group assignment minimizing squared frame-count error to 60/20/20; groups are indivisible'
} else {
    $devGroups = [math]::Floor($groups.Count * 0.60)
    $validationGroups = [math]::Floor($groups.Count * 0.20)
    for ($i = 0; $i -lt $groups.Count; $i++) {
        $splitByGroup[$groups[$i].name] = if ($i -lt $devGroups) { 'development' } elseif ($i -lt ($devGroups + $validationGroups)) { 'validation' } else { 'internal_test' }
    }
    $splitPolicy = 'stable SHA-256 ordering of scene groups; 60/20/20 by group count'
}

$frames = @($selected | Sort-Object domain,sceneGroup,selectionKey | ForEach-Object {
    [ordered]@{
        id = $_.id
        domain = $_.domain
        sceneGroup = $_.sceneGroup
        split = $splitByGroup[$_.sceneGroup]
        clear = $_.image
        depth = $_.depth
        depthMask = $_.mask
    }
})

$airlights = @(
    [ordered]@{ id = 'neutral'; rgbLinear = @(0.90, 0.90, 0.90) },
    [ordered]@{ id = 'cool'; rgbLinear = @(0.82, 0.88, 0.95) },
    [ordered]@{ id = 'warm'; rgbLinear = @(0.95, 0.88, 0.80) }
)
$noiseModels = @(
    [ordered]@{ id = 'clean'; gaussianSigma = 0.0; poissonPeak = 0.0 },
    [ordered]@{ id = 'gaussian'; gaussianSigma = 0.005; poissonPeak = 0.0 },
    [ordered]@{ id = 'poisson_gaussian'; gaussianSigma = 0.003; poissonPeak = 4096.0 }
)
$targetTransmissionAtP90Depth = @(0.80, 0.60, 0.40, 0.20, 0.10)
$recipeCount = $frames.Count * $targetTransmissionAtP90Depth.Count * $airlights.Count * $noiseModels.Count
$splitCounts = @{}
foreach ($split in @('development','validation','internal_test')) { $splitCounts[$split] = @($frames | Where-Object split -eq $split).Count }

$manifest = [ordered]@{
    schemaVersion = 1
    dataset = 'DIODE validation'
    generatedUtc = [DateTime]::UtcNow.ToString('o')
    sourceRoot = $source
    seed = $Seed
    selection = [ordered]@{
        completeTriplesFound = @($candidates).Count
        framesSelected = $frames.Count
        uniqueSceneGroups = $groups.Count
        splitPolicy = $splitPolicy
        splitFrameCounts = $splitCounts
    }
    synthesis = [ordered]@{
        colorSpace = 'linear RGB'
        model = 'I = J*t + A*(1-t)'
        depthPolicy = 'valid positive finite depth; beta=-ln(targetTransmission)/p90(validDepth); t=exp(-beta*depth)'
        targetTransmissionAtP90Depth = $targetTransmissionAtP90Depth
        airlights = $airlights
        noiseModels = $noiseModels
        materialization = 'streaming; hazy and transmission images are not persisted'
        recipeCount = $recipeCount
    }
    frames = $frames
}

$parent = [System.IO.Path]::GetDirectoryName($outputPath)
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Host "DIODE-MANIFEST $outputPath"
Write-Host "FRAMES found=$(@($candidates).Count) selected=$($frames.Count) scene_groups=$($groups.Count)"
Write-Host "SPLITS development=$($splitCounts.development) validation=$($splitCounts.validation) internal_test=$($splitCounts.internal_test)"
Write-Host "RECIPES $recipeCount (streaming; no generated images written)"

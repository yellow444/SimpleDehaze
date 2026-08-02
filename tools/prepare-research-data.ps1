param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('DIODE-Val')]
    [string]$Name,

    [string]$Destination = (Join-Path $PSScriptRoot '..\benchdata'),
    [double]$MinimumFreeReserveGiB = 10,
    [switch]$Download,
    [switch]$Extract,
    [switch]$InventoryOnly
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$catalogPath = Join-Path $repo 'datasets\data-sources.json'
$catalog = Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json
$definition = $catalog.datasets | Where-Object name -eq $Name | Select-Object -First 1
if (-not $definition) { throw "Dataset is not present in $catalogPath`: $Name" }

$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
$archiveRoot = Join-Path $destinationRoot 'archives'
$extractedRoot = Join-Path $destinationRoot 'rgbd_sources'
$archiveName = if ($Name -eq 'DIODE-Val') { 'diode_val.tar.gz' } else { $Name + '.archive' }
$archivePath = Join-Path $archiveRoot $archiveName
$partialPath = $archivePath + '.partial'
$datasetRoot = Join-Path $extractedRoot 'diode_val'
$completionMarker = Join-Path $datasetRoot '.simpledehaze-extract-complete.json'

function Get-FreeBytes([string]$path) {
    $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($path))
    return ([System.IO.DriveInfo]::new($root)).AvailableFreeSpace
}

function Assert-FreeSpace([long]$additionalBytes, [string]$phase) {
    $reserve = [long]($MinimumFreeReserveGiB * 1GB)
    $free = Get-FreeBytes $destinationRoot
    $required = $additionalBytes + $reserve
    Write-Host ("SPACE {0}: free={1:N2} GiB required={2:N2} GiB reserve={3:N2} GiB" -f $phase, ($free / 1GB), ($required / 1GB), $MinimumFreeReserveGiB)
    if ($free -lt $required) {
        throw "Not enough disk space for $phase. Free=$([math]::Round($free/1GB,2)) GiB; required including reserve=$([math]::Round($required/1GB,2)) GiB."
    }
}

function Download-Resumable([string]$uri, [string]$target, [long]$expectedBytes) {
    $part = $target + '.partial'
    $offset = if (Test-Path -LiteralPath $part) { (Get-Item -LiteralPath $part).Length } else { 0L }
    if ($offset -gt $expectedBytes) { throw "Partial download is larger than expected: $part" }
    if ($offset -eq $expectedBytes) { Move-Item -LiteralPath $part -Destination $target; return }

    $handler = [System.Net.Http.HttpClientHandler]::new()
    $client = [System.Net.Http.HttpClient]::new($handler)
    $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get, $uri)
    if ($offset -gt 0) { $request.Headers.Range = [System.Net.Http.Headers.RangeHeaderValue]::new($offset, $null) }
    try {
        $response = $client.Send($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead)
        [void]$response.EnsureSuccessStatusCode()
        $append = $offset -gt 0 -and $response.StatusCode -eq [System.Net.HttpStatusCode]::PartialContent
        if (-not $append) { $offset = 0 }
        $mode = if ($append) { [System.IO.FileMode]::Append } else { [System.IO.FileMode]::Create }
        $source = $response.Content.ReadAsStream()
        $targetStream = [System.IO.FileStream]::new($part, $mode, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None, 4MB, [System.IO.FileOptions]::SequentialScan)
        try {
            $buffer = [byte[]]::new(4MB)
            $downloaded = $offset
            $nextReport = $downloaded + 256MB
            while (($read = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $targetStream.Write($buffer, 0, $read)
                $downloaded += $read
                if ($downloaded -ge $nextReport) {
                    Write-Host ("DOWNLOAD {0:N2}/{1:N2} GiB ({2:P1})" -f ($downloaded/1GB), ($expectedBytes/1GB), ($downloaded/[double]$expectedBytes))
                    $nextReport += 256MB
                }
            }
        } finally {
            $targetStream.Dispose()
            $source.Dispose()
        }
    } finally {
        if ($response) { $response.Dispose() }
        $request.Dispose(); $client.Dispose(); $handler.Dispose()
    }
    $actual = (Get-Item -LiteralPath $part).Length
    if ($actual -ne $expectedBytes) { throw "Downloaded size mismatch: got $actual, expected $expectedBytes. Partial file kept for resume." }
    Move-Item -LiteralPath $part -Destination $target
}

function Write-Inventory {
    New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
    $files = Get-ChildItem -LiteralPath $destinationRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike '*.partial' -and $_.FullName -ne (Join-Path $destinationRoot 'files_sha256.json') }
    $items = foreach ($file in $files) {
        [ordered]@{
            path = [System.IO.Path]::GetRelativePath($destinationRoot, $file.FullName).Replace('\', '/')
            bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $inventoryPath = Join-Path $destinationRoot 'files_sha256.json'
    [ordered]@{ generated = (Get-Date).ToString('o'); root = $destinationRoot; files = @($items) } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $inventoryPath -Encoding utf8
    Write-Host "INVENTORY $inventoryPath files=$(@($items).Count)"
}

function Assert-DiodeExtractionComplete {
    $png = @(Get-ChildItem -LiteralPath $datasetRoot -Recurse -File -Filter '*.png').Count
    $depth = @(Get-ChildItem -LiteralPath $datasetRoot -Recurse -File -Filter '*_depth.npy' |
        Where-Object Name -NotLike '*_depth_mask.npy').Count
    $mask = @(Get-ChildItem -LiteralPath $datasetRoot -Recurse -File -Filter '*_depth_mask.npy').Count
    if ($png -ne 771 -or $depth -ne 771 -or $mask -ne 771) {
        throw "Incomplete DIODE extraction: png=$png depth=$depth mask=$mask; expected 771 of each. The directory is retained for diagnosis: $datasetRoot"
    }
    return [ordered]@{ png = $png; depth = $depth; masks = $mask }
}

function Write-ExtractionMarker($counts) {
    [ordered]@{
        dataset = 'DIODE-Val'
        archiveBytes = $archiveBytes
        archiveMd5 = $definition.md5.ToLowerInvariant()
        validatedUtc = [DateTime]::UtcNow.ToString('o')
        files = $counts
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $completionMarker -Encoding utf8
}

New-Item -ItemType Directory -Force -Path $archiveRoot,$extractedRoot | Out-Null
$archiveBytes = [long]$definition.archiveBytes
$existingArchiveBytes = if (Test-Path -LiteralPath $archivePath) { (Get-Item -LiteralPath $archivePath).Length } elseif (Test-Path -LiteralPath $partialPath) { (Get-Item -LiteralPath $partialPath).Length } else { 0L }

if ($Download -and -not (Test-Path -LiteralPath $archivePath)) {
    Assert-FreeSpace ([math]::Max(0L, $archiveBytes - $existingArchiveBytes)) 'download'
    Download-Resumable $definition.url $archivePath $archiveBytes
}

if (Test-Path -LiteralPath $archivePath) {
    $actualBytes = (Get-Item -LiteralPath $archivePath).Length
    if ($actualBytes -ne $archiveBytes) { throw "Archive size mismatch: $actualBytes != $archiveBytes" }
    $md5 = (Get-FileHash -LiteralPath $archivePath -Algorithm MD5).Hash.ToLowerInvariant()
    if ($definition.md5 -and $md5 -ne $definition.md5.ToLowerInvariant()) { throw "MD5 mismatch: $md5 != $($definition.md5)" }
    Write-Host "ARCHIVE-OK $archivePath bytes=$actualBytes md5=$md5"
} elseif (-not $InventoryOnly) {
    throw "Archive is absent. Use -Download: $archivePath"
}

if ($Extract) {
    if (-not (Test-Path -LiteralPath $archivePath)) { throw "Cannot extract missing archive: $archivePath" }
    $existing = if (Test-Path -LiteralPath $datasetRoot) { @(Get-ChildItem -LiteralPath $datasetRoot -Force) } else { @() }
    if ($existing.Count -eq 0) {
        Assert-FreeSpace ([long]$definition.estimatedExtractedBytes) 'extraction'
        New-Item -ItemType Directory -Force -Path $datasetRoot | Out-Null
        & tar -xzf $archivePath -C $datasetRoot
        if ($LASTEXITCODE -ne 0) { throw "tar extraction failed with code $LASTEXITCODE" }
        $counts = Assert-DiodeExtractionComplete
        Write-ExtractionMarker $counts
    } elseif (-not (Test-Path -LiteralPath $completionMarker -PathType Leaf)) {
        # A previous version of this script had no marker. Adopt it only after exact structural validation;
        # a partial tar extraction must never be silently reused.
        $counts = Assert-DiodeExtractionComplete
        Write-ExtractionMarker $counts
        Write-Host "ADOPT-VALID-EXTRACTION $datasetRoot"
    } else {
        [void](Assert-DiodeExtractionComplete)
        Write-Host "REUSE-EXTRACTED $datasetRoot"
    }
}

Write-Inventory
$freeAfter = Get-FreeBytes $destinationRoot
Write-Host ("DONE dataset={0} free_after={1:N2} GiB" -f $Name, ($freeAfter/1GB))

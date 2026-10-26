<#
.SYNOPSIS
  Start LLMServe on Windows (Docker Desktop, WSL2 backend, NVIDIA GPU): checks, builds, starts, waits for the API.

.DESCRIPTION
  Does what the README's quick start does, with the checks that save an hour of waiting:
    - Docker Desktop and the NVIDIA driver (580 or newer) are there, and the GPU is visible inside a container
    - %UserProfile%\.wslconfig gives the WSL2 VM enough RAM for the chosen quant (it only gets half of your RAM by default)
    - .env exists (made from .env.example, with a new API key if you ask for one) and CUDA_ARCHS matches your card
  then runs `docker compose up -d --build` and follows the log until the API answers.

.EXAMPLE
  .\scripts\start.ps1                       # Q2_0, 64K context
  .\scripts\start.ps1 -Quant IQ3_XXS -Context 131072 -NewKey
  .\scripts\start.ps1 -Down                 # stop (the model stays in the Docker volume)
  .\scripts\start.ps1 -Logs                 # follow the log of a running container
#>
[CmdletBinding()]
param(
  [string]$Quant,                 # Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, IQ1_M, UD-IQ4_XS, UD-Q4_K_XL ... (default: what .env says)
  [string]$Context,               # tokens, 64k, 128k, auto
  [switch]$NewKey,                # write a new random API key into .env
  [switch]$NoBuild,               # do not rebuild the image
  [switch]$Down,                  # stop and remove the container (not the model)
  [switch]$Logs,                  # just follow the log
  [switch]$SkipChecks             # skip the .wslconfig and GPU checks
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Say($m)  { Write-Host "[llmserve] $m" }
function Warn($m) { Write-Host "[llmserve] WARN  $m" -ForegroundColor Yellow }
function Fail($m) { Write-Host "[llmserve] ERROR $m" -ForegroundColor Red; exit 1 }

# RAM each quant asks for (GB), the catalog's "ram_gb"
$ramNeed = @{ 'Q2_0' = 48; 'IQ2_XS' = 48; 'IQ3_XXS' = 60; 'IQ3_S' = 62; 'IQ1_M' = 32; 'UD-IQ4_XS' = 48; 'UD-Q4_K_XL' = 48;
              'UD-Q5_K_XL' = 108; 'UD-Q6_K_XL' = 118; 'Q8_0' = 136 }

function Get-EnvValue($name) {
  if (-not (Test-Path .env)) { return $null }
  $line = Get-Content .env | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -Last 1
  if ($line) { return ($line -replace "^\s*$name\s*=\s*", '').Trim().Trim('"') }
  return $null
}
function Set-EnvValue($name, $value) {
  $lines = if (Test-Path .env) { @(Get-Content .env) } else { @() }
  $found = $false
  $out = foreach ($l in $lines) {
    if ($l -match "^\s*#?\s*$name\s*=") { if (-not $found) { $found = $true; "$name=$value" } } else { $l }
  }
  if (-not $found) { $out = @($out) + "$name=$value" }
  Set-Content -Path .env -Value $out -Encoding ASCII
}

if ($Down) { docker compose down; exit $LASTEXITCODE }
if ($Logs) { docker compose logs -f --tail 100; exit $LASTEXITCODE }

# ---- Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail 'docker was not found. Install Docker Desktop (WSL2 backend): https://www.docker.com/products/docker-desktop/' }
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail 'Docker is not running. Start Docker Desktop and wait until it says "Engine running".' }

# ---- .env
if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  Say 'created .env from .env.example'
}
if ($Quant)   { Set-EnvValue 'LLMSERVE_QUANT' $Quant }
if ($Context) { Set-EnvValue 'LLMSERVE_CONTEXT' $Context }
if ($NewKey -or -not (Get-EnvValue 'LLMSERVE_API_KEY')) {
  if ($NewKey) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = ($bytes | ForEach-Object { $_.ToString('x2') }) -join ''
    Set-EnvValue 'LLMSERVE_API_KEY' $key
    Say "new API key written to .env: $key"
  }
}
$q = Get-EnvValue 'LLMSERVE_QUANT'; if (-not $q) { $q = 'Q2_0' }

if (-not $SkipChecks) {
  # ---- NVIDIA driver and GPU
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($smi) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader | Select-Object -First 1)
    Say "GPU: $gpu"
    $parts = $gpu -split ',\s*'
    if ($parts.Count -ge 4) {
      $driverMajor = [int](($parts[2] -split '\.')[0])
      if ($driverMajor -lt 580) { Fail "NVIDIA driver $($parts[2]) is too old: the engine uses CUDA 13 and needs driver 580 or newer. Update it from nvidia.com/drivers." }
      # build only for this card unless CUDA_ARCHS is set already
      if (-not (Get-EnvValue 'CUDA_ARCHS')) {
        $arch = ($parts[3] -replace '\.', '')
        Set-EnvValue 'CUDA_ARCHS' "`"$arch-real;$arch-virtual`""
        Say "CUDA_ARCHS set to $arch for this card (a faster build; edit .env to change)"
      }
    }
  } else {
    Warn 'nvidia-smi was not found on Windows: is the NVIDIA driver installed? Continuing.'
  }

  # ---- WSL2 memory: the VM gets half of your RAM unless .wslconfig says otherwise
  $physGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
  $need = if ($ramNeed.ContainsKey($q)) { $ramNeed[$q] } else { 48 }
  $cfgPath = Join-Path $env:UserProfile '.wslconfig'
  $wslGB = [math]::Floor($physGB / 2)
  $explicit = $false
  if (Test-Path $cfgPath) {
    $m = Select-String -Path $cfgPath -Pattern '^\s*memory\s*=\s*(\d+)\s*(GB|MB|G|M)?' | Select-Object -First 1
    if ($m) {
      $explicit = $true
      $n = [double]$m.Matches[0].Groups[1].Value
      $unit = $m.Matches[0].Groups[2].Value
      $wslGB = if ($unit -match '^M') { [math]::Floor($n / 1024) } else { [math]::Floor($n) }
    }
  }
  Say "Windows RAM: $physGB GB; WSL2 VM memory: $wslGB GB$(if (-not $explicit) { ' (the default: half)' }); $q asks for $need GB"
  if ($wslGB -lt $need) {
    $rec = [math]::Floor($physGB - [math]::Max(8, $physGB / 8))   # leave Windows 8 GB, or an eighth on big PCs
    Warn "the WSL2 VM has too little RAM for $q, so it would fall back to a much slower low-RAM mode."
    Warn "Create or edit $cfgPath :"
    Write-Host "      [wsl2]"
    Write-Host "      memory=${rec}GB"
    Warn 'then run `wsl --shutdown`, restart Docker Desktop, and start again. Details: docs\WINDOWS_DOCKER.md'
    if ($physGB -lt $need + 8) { Warn "this PC has only $physGB GB of RAM: choose a smaller quant (IQ1_M needs 32 GB) or accept low-RAM mode." }
    $ans = Read-Host 'Continue anyway? [y/N]'
    if ($ans -notmatch '^[yY]') { exit 1 }
  }

  # ---- the GPU inside a container
  Say 'checking that Docker can pass the GPU through ...'
  docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi -L
  if ($LASTEXITCODE -ne 0) {
    Fail 'Docker cannot see the GPU. Use Docker Desktop with the WSL2 backend (Settings > General), update the NVIDIA driver, restart Docker Desktop. docs\WINDOWS_DOCKER.md, "Troubleshooting".'
  }
}

# ---- build and start
if ($NoBuild) { docker compose up -d } else { docker compose up -d --build }
if ($LASTEXITCODE -ne 0) { Fail 'docker compose failed (see above).' }

$port = Get-EnvValue 'LLMSERVE_HOST_PORT'; if (-not $port) { $port = '8080' }
Say "started. First start downloads the model (~70 GB for Q2_0) and packs it; this can take an hour or two."
Say "following the log (Ctrl+C leaves it running); waiting for http://localhost:$port/health ..."
$job = Start-Job { docker compose logs -f --tail 20 }
try {
  while ($true) {
    Receive-Job $job | ForEach-Object { Write-Host $_ }
    try {
      $h = Invoke-RestMethod -Uri "http://localhost:$port/health" -TimeoutSec 3
      if ($h.status -eq 'ok') { break }
    } catch { }
    $state = (docker inspect -f '{{.State.Status}}' llmserve 2>$null)
    if ($state -and $state -ne 'running' -and $state -ne 'restarting') { Fail "the container stopped ($state). Read the log above: docker compose logs" }
    Start-Sleep -Seconds 5
  }
} finally {
  Stop-Job $job -ErrorAction SilentlyContinue; Remove-Job $job -Force -ErrorAction SilentlyContinue
}
Say "ready: http://localhost:$port/v1  (model: $($h.model), context $($h.max_context))"
Say 'try:  Invoke-RestMethod http://localhost:8080/v1/models'

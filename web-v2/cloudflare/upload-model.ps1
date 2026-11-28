param(
  [string]$Remote = "r2",
  [string]$Bucket = "safevichi-models",
  [string]$OnnxDir = ""
)

$ErrorActionPreference = "Stop"
$webRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $webRoot
if (-not $OnnxDir) {
  $OnnxDir = Join-Path $repoRoot "models\dualorc_mmbert_base\onnx"
}

$manifestPath = Join-Path $OnnxDir "export_manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
  throw "Không tìm thấy $manifestPath. Hãy chạy 'python -m src.v2.download_model' trước."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$int8Sha = $manifest.sha256.onnx_int8
$crfSha = $manifest.sha256.crf_params
if (-not $int8Sha -or $int8Sha.Length -lt 12) {
  throw "export_manifest.json thiếu sha256.onnx_int8."
}

$version = "safevichi-v2-int8-$($int8Sha.Substring(0, 12))"
$destinationRoot = "{0}:{1}/models/{2}" -f $Remote, $Bucket, $version

$files = @(
  @{ Rel = "safevichi_v2_int8.onnx"; ContentType = "application/octet-stream"; ExpectedSha = $int8Sha },
  @{ Rel = "crf_params.json"; ContentType = "application/json; charset=utf-8"; ExpectedSha = $crfSha },
  @{ Rel = "export_manifest.json"; ContentType = "application/json; charset=utf-8"; ExpectedSha = $null },
  @{ Rel = "tokenizer/tokenizer.json"; ContentType = "application/json; charset=utf-8"; ExpectedSha = $null },
  @{ Rel = "tokenizer/tokenizer_config.json"; ContentType = "application/json; charset=utf-8"; ExpectedSha = $null },
  @{ Rel = "tokenizer/backbone_config.json"; ContentType = "application/json; charset=utf-8"; ExpectedSha = $null }
)

foreach ($item in $files) {
  $source = Join-Path $OnnxDir ($item.Rel -replace '/', '\')
  if (-not (Test-Path -LiteralPath $source)) {
    throw "Thiếu tệp mô hình: $source"
  }
  $resolvedSource = (Resolve-Path -LiteralPath $source).Path
  if ($item.ExpectedSha) {
    $actualSha = (Get-FileHash -LiteralPath $resolvedSource -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha -ne $item.ExpectedSha.ToLowerInvariant()) {
      throw "$($item.Rel) không khớp SHA-256 trong export_manifest.json."
    }
  }

  $target = "$destinationRoot/$($item.Rel)"
  Write-Host "Uploading $($item.Rel) -> $target"
  rclone copyto $resolvedSource $target `
    --progress `
    --s3-upload-cutoff 100M `
    --s3-chunk-size 100M `
    --header-upload "Content-Type: $($item.ContentType)" `
    --header-upload "Cache-Control: public, max-age=31536000, immutable"
  if ($LASTEXITCODE -ne 0) {
    throw "rclone thất bại khi tải lên $($item.Rel)."
  }
}

Write-Host "Uploaded SafeViChi v2 immutable model package ($version) to $destinationRoot"
Write-Host "Set VITE_MODEL_BASE_URL to: https://<r2-public-host>/models/$version"

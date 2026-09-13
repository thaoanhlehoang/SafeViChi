param(
  [string]$Remote = "r2",
  [string]$Bucket = "safevichi-models"
)

$ErrorActionPreference = "Stop"
$demoRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $demoRoot "model-manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$modelDirectory = Join-Path $demoRoot "onnx_model\onnx"
$destinationRoot = "{0}:{1}/models/{2}" -f $Remote, $Bucket, $manifest.version

foreach ($name in @("encoder", "decoder")) {
  $asset = $manifest.assets.$name
  $source = Join-Path $modelDirectory $asset.path
  $resolvedSource = (Resolve-Path -LiteralPath $source).Path
  $length = (Get-Item -LiteralPath $resolvedSource).Length
  $sha256 = (Get-FileHash -LiteralPath $resolvedSource -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($length -ne $asset.bytes -or $sha256 -ne $asset.sha256) {
    throw "$name does not match model-manifest.json."
  }

  rclone copyto $resolvedSource "$destinationRoot/$($asset.path)" --immutable --progress --s3-upload-cutoff 100M --s3-chunk-size 100M --header-upload "Content-Type: application/octet-stream" --header-upload "Cache-Control: public, max-age=31536000, immutable"
  if ($LASTEXITCODE -ne 0) { throw "rclone failed while uploading $name." }
}

Write-Host "Uploaded immutable model version $($manifest.version) to $destinationRoot"

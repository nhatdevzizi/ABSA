param(
    [ValidateSet("tune", "train", "evaluate")][string]$Command = "train",
    [ValidateSet("all", "TF_LR", "TF_SVM")][string]$Model = "all"
)
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    uv run python -m uit_vsfc $Command --model $Model
    if ($LASTEXITCODE -ne 0) { throw "Baseline command failed ($LASTEXITCODE)" }
} finally { Pop-Location }

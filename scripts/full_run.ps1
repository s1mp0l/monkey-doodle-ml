# Полное обучение классификатора рисунков — одной командой, с логом и остановкой на первой ошибке.
#   powershell -ExecutionPolicy Bypass -File scripts\full_run.ps1
# Синтетика/аугментация/композиции уже на диске (data/synthetic, data/augmented). Пересобрать их
# можно флагом -Regen. Данные Quick Draw докачиваются (download.py резюмируемый).
param(
  [int]$Epochs = 30,
  [int]$Batch = 512,
  [switch]$Regen,          # заново сгенерировать синтетику/аугментацию/композиции
  [switch]$SkipDownload    # пропустить скачивание Quick Draw (если уже скачано)
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $root "runs\$ts"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$log = Join-Path $runDir "full_run.log"

function Step($name, [scriptblock]$cmd) {
  "=== $name  $(Get-Date -Format o) ===" | Tee-Object -FilePath $log -Append
  & $cmd 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) { "ОШИБКА на шаге «$name» (код $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
}

Step "проверка CUDA" { & $py -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))" }
if (-not $SkipDownload) { Step "скачивание Quick Draw (full)" { & $py scripts/download.py } }
if ($Regen) {
  Step "калибровка + синтетика rude" { & $py scripts/synth_rude.py --calibrate data/custom/rude.ndjson -n 6000 --seed 1 --shapes swastika=2500,butt=1000,breasts=1000 }
  Step "аугментация + композиции"    { & $py scripts/augment_rude.py }
}
Step "подготовка данных (full)"      { & $py scripts/prepare.py --profile full }
Step "обучение"                      { & $py scripts/train.py --data data/prepared/full --epochs $Epochs --batch $Batch --out $runDir }
Step "оценка"                        { & $py scripts/evaluate.py --run $runDir }
Step "экспорт ONNX"                  { & $py scripts/export_onnx.py --run $runDir }
"ГОТОВО. Прогон: $runDir" | Tee-Object -FilePath $log -Append
"Смотрите $runDir\eval\report.md (rude на людях, подтипы, композиции, путаницы) и models\latest\." | Tee-Object -FilePath $log -Append

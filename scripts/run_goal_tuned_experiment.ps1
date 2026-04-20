param(
    [string]$CondaEnv = "dl_gpu",
    [string]$PpoConfig = "config/ppo_config.yaml",
    [string]$RewardConfig = "config/reward_config.yaml",
    [string]$ModelConfig = "config/model_config.yaml",
    [string]$TrainSeeds = "11,22,33,44,55",
    [string]$EvalSeeds = "42,43,44",
    [string]$Modes = "greedy,sampling",
    [string]$TrainOutputRoot = "models/policy_ppo_multiseed_goal_tuned",
    [string]$EvalOutputRoot = "results/multiseed_eval_goal_tuned",
    [string]$SftPath = "models/policy_sft",
    [string]$PpoPathOverride = "",
    [string]$SelectModel = "best",
    [bool]$IncludeBase = $true,
    [switch]$LiveOutput,
    [switch]$ShowTqdm,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [string]$RunLogRoot = "results/run_logs"
)

$ErrorActionPreference = "Stop"

try {
    chcp 65001 > $null
} catch {
}

try {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $OutputEncoding = $utf8NoBom
} catch {
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:TQDM_ASCII = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:HF_DATASETS_DISABLE_PROGRESS_BARS = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:TRANSFORMERS_VERBOSITY = "error"

if ($ShowTqdm) {
    $env:RLHF_DISABLE_TQDM = "0"
    $env:RLHF_TQDM_MININTERVAL = "0.5"
} else {
    # Default to low-redraw mode to reduce VS Code terminal freezes on Windows.
    $env:RLHF_DISABLE_TQDM = "1"
    $env:RLHF_TQDM_MININTERVAL = "1.0"
}

$scriptStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$runLogDir = Join-Path $RunLogRoot (Get-Date -Format "yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force -Path $runLogDir | Out-Null

function Get-StepLogPath {
    param([string]$StepName)

    $safeStepName = ($StepName -replace "[^A-Za-z0-9._-]", "_")
    return Join-Path $runLogDir ("{0}.log" -f $safeStepName)
}

function Split-SeedList {
    param([string]$Raw)
    return $Raw.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
}

function Invoke-CondaPython {
    param(
        [string[]]$PythonArgs,
        [string]$StepName
    )

    if ($LiveOutput) {
        $cmd = @("run", "--no-capture-output", "-n", $CondaEnv, "python") + $PythonArgs
    } else {
        $cmd = @("run", "-n", $CondaEnv, "python") + $PythonArgs
    }

    $stepLogPath = Get-StepLogPath -StepName $StepName
    $stepStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host ""
    Write-Host "[$StepName]" -ForegroundColor Cyan
    Write-Host ("conda " + ($cmd -join " ")) -ForegroundColor DarkGray
    Write-Host "log file: $stepLogPath" -ForegroundColor DarkGray

    if ($LiveOutput) {
        & conda @cmd 2>&1 | Tee-Object -FilePath $stepLogPath
    } else {
        & conda @cmd *> $stepLogPath
    }

    $stepStopwatch.Stop()
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $stepLogPath) {
            Write-Host ""
            Write-Host "Last 40 log lines:" -ForegroundColor Yellow
            Get-Content -Path $stepLogPath -Tail 40 | ForEach-Object { Write-Host $_ }
        }
        throw "Step '$StepName' failed with exit code $LASTEXITCODE"
    }

    if ((-not $LiveOutput) -and (Test-Path $stepLogPath)) {
        Write-Host "Recent log lines:" -ForegroundColor DarkGray
        Get-Content -Path $stepLogPath -Tail 5 | ForEach-Object { Write-Host ("  " + $_) }
    }

    Write-Host ("[{0}] finished in {1:n1}s" -f $StepName, $stepStopwatch.Elapsed.TotalSeconds) -ForegroundColor Green
}

function Get-BestSeed {
    param(
        [string[]]$Seeds,
        [string]$Root
    )

    $bestSeed = $null
    $bestScore = [double]::NegativeInfinity

    foreach ($seed in $Seeds) {
        $summaryPath = Join-Path (Join-Path $Root "seed_$seed") "training_summary.json"
        if (-not (Test-Path $summaryPath)) {
            continue
        }

        $summary = Get-Content -Raw -Path $summaryPath | ConvertFrom-Json
        $score = [double]$summary.best_score

        if ($score -gt $bestScore) {
            $bestScore = $score
            $bestSeed = $seed
        }
    }

    if ($null -eq $bestSeed) {
        throw "No valid training_summary.json found under $Root"
    }

    return @{ Seed = $bestSeed; Score = $bestScore }
}

if (-not (Test-Path $PpoConfig)) { throw "Missing PPO config: $PpoConfig" }
if (-not (Test-Path $RewardConfig)) { throw "Missing reward config: $RewardConfig" }
if (-not (Test-Path $ModelConfig)) { throw "Missing model config: $ModelConfig" }
if (-not (Test-Path $SftPath)) { Write-Warning "SFT path not found: $SftPath" }
if ($SelectModel -notin @("best", "final")) { throw "SelectModel must be 'best' or 'final'" }

$trainSeedList = Split-SeedList -Raw $TrainSeeds
$evalSeedList = Split-SeedList -Raw $EvalSeeds

Write-Host "============================================"
Write-Host "Goal-Tuned Experiment Runner"
Write-Host "============================================"
Write-Host "Conda env: $CondaEnv"
Write-Host "Train seeds: $($trainSeedList -join ',')"
Write-Host "Eval seeds: $($evalSeedList -join ',')"
Write-Host "Train output root: $TrainOutputRoot"
Write-Host "Eval output root: $EvalOutputRoot"
if ($LiveOutput) {
    Write-Host "Output mode: live stream" -ForegroundColor Yellow
} else {
    Write-Host "Output mode: stable file logging" -ForegroundColor Green
}
Write-Host "Run logs root: $runLogDir"

if (-not $SkipTrain) {
    New-Item -ItemType Directory -Force -Path $TrainOutputRoot | Out-Null

    foreach ($seed in $trainSeedList) {
        $seedOutDir = Join-Path $TrainOutputRoot "seed_$seed"
        New-Item -ItemType Directory -Force -Path $seedOutDir | Out-Null

        Invoke-CondaPython -StepName "Train seed $seed" -PythonArgs @(
            "-m", "src.training.train_ppo",
            "--config", $PpoConfig,
            "--reward_config", $RewardConfig,
            "--seed", $seed,
            "--save_dir", $seedOutDir
        )
    }
}

if (-not $SkipEval) {
    $ppoModelPath = $PpoPathOverride
    if ([string]::IsNullOrWhiteSpace($ppoModelPath)) {
        $best = Get-BestSeed -Seeds $trainSeedList -Root $TrainOutputRoot
        $ppoModelPath = Join-Path (Join-Path $TrainOutputRoot "seed_$($best.Seed)") $SelectModel
        Write-Host ""
        Write-Host "Selected PPO model for evaluation:" -ForegroundColor Green
        Write-Host "  seed: $($best.Seed)"
        Write-Host "  best_score: $($best.Score)"
        Write-Host "  path: $ppoModelPath"
    } else {
        Write-Host ""
        Write-Host "Using PPO override path for evaluation: $ppoModelPath" -ForegroundColor Yellow
    }

    New-Item -ItemType Directory -Force -Path $EvalOutputRoot | Out-Null

    foreach ($seed in $evalSeedList) {
        $seedOutDir = Join-Path $EvalOutputRoot "seed_$seed"
        New-Item -ItemType Directory -Force -Path $seedOutDir | Out-Null

        $evalArgs = @(
            "-m", "src.evaluation.evaluate",
            "--output", $seedOutDir,
            "--config", $ModelConfig,
            "--reward_config", $RewardConfig,
            "--seed", $seed,
            "--modes", $Modes,
            "--sft", $SftPath,
            "--ppo", $ppoModelPath
        )

        if ($IncludeBase) {
            $evalArgs += "--base"
        }

        Invoke-CondaPython -StepName "Eval seed $seed" -PythonArgs $evalArgs
    }

    $aggOut = Join-Path $EvalOutputRoot "aggregate.json"
    Invoke-CondaPython -StepName "Aggregate eval seeds" -PythonArgs @(
        "-m", "src.evaluation.aggregate_multiseed",
        "--input_root", $EvalOutputRoot,
        "--output", $aggOut
    )

    Write-Host ""
    Write-Host "Aggregate result saved to: $aggOut" -ForegroundColor Green
}

Write-Host ""
Write-Host "All requested steps completed." -ForegroundColor Green
$scriptStopwatch.Stop()
Write-Host ("Total pipeline runtime: {0:n1}s" -f $scriptStopwatch.Elapsed.TotalSeconds) -ForegroundColor Green

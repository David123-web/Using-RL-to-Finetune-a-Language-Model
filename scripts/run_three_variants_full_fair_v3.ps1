param(
    [string]$CondaEnv = "dl_gpu",
    [string]$TrainSeeds = "11,22,33,44,55",
    [string]$EvalSeeds = "42,43,44",
    [string]$EvalConfig = "config/model_config.yaml",
    [string]$EvalRewardConfig = "config/reward_config_qualityfix_short_v3.yaml",
    [string]$OutputRoot = "results/eval_fair_v3_full_three_variants",
    [string]$RunLogRoot = "results/run_logs",
    [switch]$LiveOutput
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
$env:RLHF_DISABLE_TQDM = "1"
$env:RLHF_TQDM_MININTERVAL = "1.0"

function Split-SeedList {
    param([object]$Raw)

    if ($null -eq $Raw) {
        return @()
    }

    if ($Raw -is [array]) {
        $joined = ($Raw | ForEach-Object { "$_" }) -join " "
    }
    else {
        $joined = "$Raw"
    }

    $joined = $joined -replace ",", " "
    return $joined -split "\s+" | Where-Object { $_ -ne "" }
}

function Get-StepLogPath {
    param(
        [string]$Dir,
        [string]$StepName
    )

    $safeStepName = ($StepName -replace "[^A-Za-z0-9._-]", "_")
    return Join-Path $Dir ("{0}.log" -f $safeStepName)
}

function Invoke-CondaPython {
    param(
        [string]$CondaEnv,
        [string[]]$PythonArgs,
        [string]$StepName,
        [string]$LogDir,
        [bool]$LiveOutput
    )

    if ($LiveOutput) {
        $cmd = @("run", "--no-capture-output", "-n", $CondaEnv, "python") + $PythonArgs
    }
    else {
        $cmd = @("run", "-n", $CondaEnv, "python") + $PythonArgs
    }

    $stepLogPath = Get-StepLogPath -Dir $LogDir -StepName $StepName
    $stepSw = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Host ""
    Write-Host "[$StepName]" -ForegroundColor Cyan
    Write-Host ("conda " + ($cmd -join " ")) -ForegroundColor DarkGray
    Write-Host "log file: $stepLogPath" -ForegroundColor DarkGray

    if ($LiveOutput) {
        & conda @cmd 2>&1 | Tee-Object -FilePath $stepLogPath
    }
    else {
        & conda @cmd *> $stepLogPath
    }

    $stepSw.Stop()
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path $stepLogPath) {
            Write-Host ""
            Write-Host "Last 50 log lines:" -ForegroundColor Yellow
            Get-Content -Path $stepLogPath -Tail 50 | ForEach-Object { Write-Host $_ }
        }
        throw "Step '$StepName' failed with exit code $LASTEXITCODE"
    }

    if ((-not $LiveOutput) -and (Test-Path $stepLogPath)) {
        Write-Host "Recent log lines:" -ForegroundColor DarkGray
        Get-Content -Path $stepLogPath -Tail 5 | ForEach-Object { Write-Host ("  " + $_) }
    }

    Write-Host ("[{0}] finished in {1:n1}s" -f $StepName, $stepSw.Elapsed.TotalSeconds) -ForegroundColor Green
}

function Get-Rate {
    param([int]$count, [int]$total)
    if ($total -le 0) {
        return 0.0
    }
    return [double]$count / [double]$total
}

function Get-RewardNorm {
    param([double]$rawReward)
    $low = 0.2
    $high = 0.8
    $x = ($rawReward - $low) / [math]::Max($high - $low, 1e-8)
    return [double]([math]::Min([math]::Max($x, 0.0), 1.0))
}

function Get-FBeta {
    param([double]$p, [double]$r, [double]$beta = 1.3)
    $beta2 = $beta * $beta
    $den = ($beta2 * $p) + $r
    if ($den -le 1e-12) {
        return 0.0
    }
    return ((1.0 + $beta2) * $p * $r) / $den
}

function Get-Stats {
    param($values)

    $arr = @($values | ForEach-Object { [double]$_ })
    if ($arr.Count -eq 0) {
        return [pscustomobject]@{ n = 0; mean = 0.0; std = 0.0; min = 0.0; max = 0.0 }
    }

    $mean = [double](($arr | Measure-Object -Average).Average)
    $min = [double](($arr | Measure-Object -Minimum).Minimum)
    $max = [double](($arr | Measure-Object -Maximum).Maximum)

    if ($arr.Count -le 1) {
        $std = 0.0
    }
    else {
        $sumSq = 0.0
        foreach ($v in $arr) {
            $sumSq += [math]::Pow($v - $mean, 2)
        }
        $std = [math]::Sqrt($sumSq / [double]($arr.Count - 1))
    }

    return [pscustomobject]@{
        n = $arr.Count
        mean = [math]::Round($mean, 4)
        std = [math]::Round($std, 4)
        min = [math]::Round($min, 4)
        max = [math]::Round($max, 4)
    }
}

$scriptSw = [System.Diagnostics.Stopwatch]::StartNew()
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLogDir = Join-Path $RunLogRoot ("three_variants_full_fair_v3_" + $timestamp)
New-Item -ItemType Directory -Force -Path $runLogDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$trainSeedList = Split-SeedList -Raw $TrainSeeds
$evalSeedList = Split-SeedList -Raw $EvalSeeds

$variants = @(
    @{
        Name = "old_goal"
        Train = $false
        PpoConfig = ""
        TrainRewardConfig = ""
        ModelRoot = "models/policy_ppo_multiseed_goal_tuned_try_now_rerun"
    },
    @{
        Name = "v1"
        Train = $true
        PpoConfig = "config/ppo_config_qualityfix_v1_full.yaml"
        TrainRewardConfig = "config/reward_config_qualityfix_short.yaml"
        ModelRoot = "models/policy_ppo_qualityfix_v1_full"
    },
    @{
        Name = "f1"
        Train = $true
        PpoConfig = "config/ppo_config_qualityfix_f1_full.yaml"
        TrainRewardConfig = "config/reward_config_qualityfix_short_f1.yaml"
        ModelRoot = "models/policy_ppo_qualityfix_f1_full"
    }
)

Write-Host "============================================"
Write-Host "Three-Variant Full Training + Fair V3 Eval"
Write-Host "============================================"
Write-Host "Conda env: $CondaEnv"
Write-Host "Train seeds: $($trainSeedList -join ',')"
Write-Host "Eval seeds: $($evalSeedList -join ',')"
Write-Host "Eval config: $EvalConfig"
Write-Host "Eval reward config (fair v3): $EvalRewardConfig"
Write-Host "Output root: $OutputRoot"
Write-Host "Run logs: $runLogDir"

foreach ($variant in $variants) {
    if ($variant.Train) {
        New-Item -ItemType Directory -Force -Path $variant.ModelRoot | Out-Null

        foreach ($seed in $trainSeedList) {
            $seedDir = Join-Path $variant.ModelRoot ("seed_" + $seed)
            $bestDir = Join-Path $seedDir "best"

            if (Test-Path $bestDir) {
                Write-Host "[SKIP][train] $($variant.Name) seed=$seed (best exists)" -ForegroundColor Yellow
                continue
            }

            New-Item -ItemType Directory -Force -Path $seedDir | Out-Null
            Invoke-CondaPython -CondaEnv $CondaEnv -LiveOutput:$LiveOutput -LogDir $runLogDir -StepName ("train_" + $variant.Name + "_seed_" + $seed) -PythonArgs @(
                "-m", "src.training.train_ppo",
                "--config", $variant.PpoConfig,
                "--reward_config", $variant.TrainRewardConfig,
                "--seed", $seed,
                "--save_dir", $seedDir
            )
        }
    }
}

foreach ($variant in $variants) {
    foreach ($trainSeed in $trainSeedList) {
        $modelPath = Join-Path (Join-Path $variant.ModelRoot ("seed_" + $trainSeed)) "best"
        if (!(Test-Path $modelPath)) {
            throw "Missing model path for eval: $modelPath"
        }

        foreach ($evalSeed in $evalSeedList) {
            $evalOut = Join-Path $OutputRoot ($variant.Name + "/train_seed_" + $trainSeed + "/seed_" + $evalSeed)
            New-Item -ItemType Directory -Force -Path $evalOut | Out-Null

            Invoke-CondaPython -CondaEnv $CondaEnv -LiveOutput:$LiveOutput -LogDir $runLogDir -StepName ("eval_" + $variant.Name + "_trainseed_" + $trainSeed + "_evalseed_" + $evalSeed) -PythonArgs @(
                "-m", "src.evaluation.evaluate",
                "--ppo", $modelPath,
                "--config", $EvalConfig,
                "--reward_config", $EvalRewardConfig,
                "--seed", $evalSeed,
                "--modes", "greedy,sampling",
                "--output", $evalOut
            )
        }
    }
}

$rows = @()
foreach ($variant in $variants) {
    foreach ($trainSeed in $trainSeedList) {
        foreach ($evalSeed in $evalSeedList) {
            $evalOut = Join-Path $OutputRoot ($variant.Name + "/train_seed_" + $trainSeed + "/seed_" + $evalSeed)
            $resultPath = Join-Path $evalOut "ppo_results.json"
            $greedyRecordsPath = Join-Path $evalOut "ppo_greedy_records.jsonl"
            $samplingRecordsPath = Join-Path $evalOut "ppo_sampling_records.jsonl"

            if (!(Test-Path $resultPath) -or !(Test-Path $greedyRecordsPath) -or !(Test-Path $samplingRecordsPath)) {
                throw "Missing eval artifact under: $evalOut"
            }

            $result = Get-Content $resultPath -Encoding UTF8 -Raw | ConvertFrom-Json
            $greedyRecords = Get-Content $greedyRecordsPath -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json }
            $samplingRecords = Get-Content $samplingRecordsPath -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json }

            $ng = @($greedyRecords).Count
            $ns = @($samplingRecords).Count
            $gQ0 = @($greedyRecords | Where-Object { [double]$_.quality_anchor -le 1e-8 }).Count
            $sQ0 = @($samplingRecords | Where-Object { [double]$_.quality_anchor -le 1e-8 }).Count

            $gRaw = [double]$result.greedy.raw_reward.mean
            $sRaw = [double]$result.sampling.raw_reward.mean
            $gQNZ = 1.0 - (Get-Rate $gQ0 $ng)
            $sQNZ = 1.0 - (Get-Rate $sQ0 $ns)

            $gFBeta = Get-FBeta (Get-RewardNorm $gRaw) $gQNZ 1.3
            $sFBeta = Get-FBeta (Get-RewardNorm $sRaw) $sQNZ 1.3
            $avgFBeta = ($gFBeta + $sFBeta) / 2.0
            $constrained = (($gRaw + $sRaw) / 2.0) * [math]::Min($gQNZ, $sQNZ)

            $rows += [pscustomobject]@{
                variant = $variant.Name
                train_seed = [int]$trainSeed
                eval_seed = [int]$evalSeed
                greedy_raw = [math]::Round($gRaw, 4)
                greedy_q_nonzero = [math]::Round($gQNZ, 4)
                sampling_raw = [math]::Round($sRaw, 4)
                sampling_q_nonzero = [math]::Round($sQNZ, 4)
                greedy_eos = [math]::Round([double]$result.greedy.eos_rate, 4)
                sampling_eos = [math]::Round([double]$result.sampling.eos_rate, 4)
                avg_fbeta_13 = [math]::Round($avgFBeta, 4)
                constrained_score = [math]::Round($constrained, 4)
            }
        }
    }
}

$summaryByVariant = @{}
foreach ($group in ($rows | Group-Object variant)) {
    $items = @($group.Group)
    $summaryByVariant[$group.Name] = [pscustomobject]@{
        n = $items.Count
        greedy_raw = Get-Stats ($items | ForEach-Object { $_.greedy_raw })
        greedy_q_nonzero = Get-Stats ($items | ForEach-Object { $_.greedy_q_nonzero })
        sampling_raw = Get-Stats ($items | ForEach-Object { $_.sampling_raw })
        sampling_q_nonzero = Get-Stats ($items | ForEach-Object { $_.sampling_q_nonzero })
        greedy_eos = Get-Stats ($items | ForEach-Object { $_.greedy_eos })
        sampling_eos = Get-Stats ($items | ForEach-Object { $_.sampling_eos })
        avg_fbeta_13 = Get-Stats ($items | ForEach-Object { $_.avg_fbeta_13 })
        constrained_score = Get-Stats ($items | ForEach-Object { $_.constrained_score })
    }
}

$summary = [pscustomobject]@{
    run_log_root = $runLogDir
    output_root = $OutputRoot
    train_seeds = @($trainSeedList | ForEach-Object { [int]$_ })
    eval_seeds = @($evalSeedList | ForEach-Object { [int]$_ })
    eval_config = $EvalConfig
    eval_reward_config = $EvalRewardConfig
    rows = $rows
    summary_by_variant = $summaryByVariant
}

$summaryPath = Join-Path $OutputRoot "summary.json"
$summary | ConvertTo-Json -Depth 10 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host ""
Write-Host "All done." -ForegroundColor Green
Write-Host "Summary: $summaryPath" -ForegroundColor Green
Write-Host "Run logs: $runLogDir" -ForegroundColor Green
$scriptSw.Stop()
Write-Host ("Total runtime: {0:n1}s" -f $scriptSw.Elapsed.TotalSeconds) -ForegroundColor Green

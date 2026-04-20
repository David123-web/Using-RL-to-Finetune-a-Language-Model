param(
    [int[]]$TrainSeeds = @(11, 22, 33),
    [int]$EvalSeed = 44,
    [string]$EvalConfig = "config/model_config_qualityfix_short.yaml",
    [string]$EvalRewardConfig = "config/reward_config_qualityfix_short_v3.yaml",
    [string]$OutputRoot = "results/eval_fair_v3metric_v1_vs_f1"
)

$ErrorActionPreference = "Stop"

function Get-Rate([int]$count, [int]$total) {
    if ($total -le 0) {
        return 0.0
    }
    return [double]$count / [double]$total
}

function Get-Stats($values) {
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

function Reward-Norm([double]$rawReward, [double]$low = 0.2, [double]$high = 0.8) {
    $lo = [math]::Min($low, $high)
    $hi = [math]::Max($low, $high)
    $denom = [math]::Max($hi - $lo, 1e-8)
    $x = ($rawReward - $lo) / $denom
    return [double]([math]::Min([math]::Max($x, 0.0), 1.0))
}

function F-Beta([double]$p, [double]$r, [double]$beta = 1.3) {
    $b2 = $beta * $beta
    $den = ($b2 * $p) + $r
    if ($den -le 1e-12) {
        return 0.0
    }
    return ((1.0 + $b2) * $p * $r) / $den
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runLogRoot = Join-Path "results/run_logs" ("v1_vs_f1_multiseed_" + $timestamp)
New-Item -ItemType Directory -Path $runLogRoot -Force | Out-Null
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:RLHF_DISABLE_TQDM = "1"

$variants = @(
    @{
        Name = "v1"
        PpoConfig = "config/ppo_config_qualityfix_short.yaml"
        RewardConfig = "config/reward_config_qualityfix_short.yaml"
        ModelRoot = "models/policy_ppo_qualityfix_short"
    },
    @{
        Name = "f1"
        PpoConfig = "config/ppo_config_qualityfix_short_f1.yaml"
        RewardConfig = "config/reward_config_qualityfix_short_f1.yaml"
        ModelRoot = "models/policy_ppo_qualityfix_short_f1"
    }
)

Write-Host "Run logs: $runLogRoot"
Write-Host "Output root: $OutputRoot"

foreach ($variant in $variants) {
    foreach ($seed in $TrainSeeds) {
        $seedDir = Join-Path $variant.ModelRoot ("seed_" + $seed)
        $bestDir = Join-Path $seedDir "best"

        if (Test-Path $bestDir) {
            Write-Host "[SKIP][train] $($variant.Name) seed=$seed (best checkpoint exists)"
        }
        else {
            $trainLog = Join-Path $runLogRoot ("train_" + $variant.Name + "_seed_" + $seed + ".log")
            Write-Host "[RUN][train] $($variant.Name) seed=$seed"
            conda run -n dl_gpu python -m src.training.train_ppo --config $variant.PpoConfig --reward_config $variant.RewardConfig --seed $seed --save_dir $seedDir *> $trainLog
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Training failed: $trainLog"
                Get-Content $trainLog -Tail 120
                throw "Training failed for $($variant.Name) seed=$seed"
            }
        }

        $evalOut = Join-Path $OutputRoot ($variant.Name + "/train_seed_" + $seed + "/seed_" + $EvalSeed)
        New-Item -ItemType Directory -Path $evalOut -Force | Out-Null

        $evalLog = Join-Path $runLogRoot ("eval_" + $variant.Name + "_trainseed_" + $seed + "_evalseed_" + $EvalSeed + ".log")
        Write-Host "[RUN][eval] $($variant.Name) train_seed=$seed eval_seed=$EvalSeed"
        conda run -n dl_gpu python -m src.evaluation.evaluate --ppo $bestDir --config $EvalConfig --reward_config $EvalRewardConfig --seed $EvalSeed --modes "greedy,sampling" --output $evalOut *> $evalLog
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Evaluation failed: $evalLog"
            Get-Content $evalLog -Tail 120
            throw "Evaluation failed for $($variant.Name) train_seed=$seed"
        }
    }
}

$rows = @()
foreach ($variant in $variants) {
    foreach ($seed in $TrainSeeds) {
        $evalOut = Join-Path $OutputRoot ($variant.Name + "/train_seed_" + $seed + "/seed_" + $EvalSeed)
        $resultPath = Join-Path $evalOut "ppo_results.json"
        $greedyRecordsPath = Join-Path $evalOut "ppo_greedy_records.jsonl"
        $samplingRecordsPath = Join-Path $evalOut "ppo_sampling_records.jsonl"

        if (!(Test-Path $resultPath) -or !(Test-Path $greedyRecordsPath) -or !(Test-Path $samplingRecordsPath)) {
            Write-Host "[WARN] Missing outputs for $($variant.Name) seed=$seed"
            continue
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

        $gFBeta = F-Beta (Reward-Norm $gRaw) $gQNZ 1.3
        $sFBeta = F-Beta (Reward-Norm $sRaw) $sQNZ 1.3
        $avgFBeta = ($gFBeta + $sFBeta) / 2.0

        $constrained = (($gRaw + $sRaw) / 2.0) * [math]::Min($gQNZ, $sQNZ)

        $rows += [pscustomobject]@{
            variant = $variant.Name
            train_seed = $seed
            eval_seed = $EvalSeed
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

$summaryByVariant = @{}
foreach ($group in ($rows | Group-Object variant)) {
    $items = @($group.Group)
    $summaryByVariant[$group.Name] = [pscustomobject]@{
        n = $items.Count
        greedy_raw = Get-Stats ($items | ForEach-Object { $_.greedy_raw })
        greedy_q_nonzero = Get-Stats ($items | ForEach-Object { $_.greedy_q_nonzero })
        sampling_raw = Get-Stats ($items | ForEach-Object { $_.sampling_raw })
        sampling_q_nonzero = Get-Stats ($items | ForEach-Object { $_.sampling_q_nonzero })
        avg_fbeta_13 = Get-Stats ($items | ForEach-Object { $_.avg_fbeta_13 })
        constrained_score = Get-Stats ($items | ForEach-Object { $_.constrained_score })
    }
}

$summary = [pscustomobject]@{
    run_log_root = $runLogRoot
    output_root = $OutputRoot
    eval_seed = $EvalSeed
    train_seeds = $TrainSeeds
    rows = $rows
    summary_by_variant = $summaryByVariant
}

$summaryPath = Join-Path $OutputRoot ("summary_eval_seed_" + $EvalSeed + ".json")
$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host "Completed. Summary: $summaryPath"
Write-Host "Run logs: $runLogRoot"

param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [Parameter(Mandatory = $true)][string]$Answers,
    [Parameter(Mandatory = $true)][string]$OfficialEvaluator,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$EncryptedKey
)

$ErrorActionPreference = 'Stop'
$projectPython = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv/Scripts/python.exe'
$datasetPath = (Resolve-Path -LiteralPath $Dataset).Path
$answersPath = (Resolve-Path -LiteralPath $Answers).Path
$evaluatorPath = (Resolve-Path -LiteralPath $OfficialEvaluator).Path
$outputPath = (Resolve-Path -LiteralPath $Output).Path
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
$keyPath = (Resolve-Path -LiteralPath $EncryptedKey).Path

$previousOpenAIKey = $env:OPENAI_API_KEY
$secure = Get-Content -LiteralPath $keyPath -Raw | ConvertTo-SecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $secure).Password
Remove-Variable secure
$previousPythonUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = '1'

try {
    foreach ($arm in @('baseline', 'memory')) {
        $chunks = Get-ChildItem -LiteralPath $outputPath -Filter "$arm-*.jsonl" | Sort-Object Name
        if ($chunks.Count -eq 0) { throw "No $arm hypothesis chunks found" }
        foreach ($chunk in $chunks) {
            $result = "$($chunk.FullName).eval-results-gpt-4o"
            $expected = (Get-Content -LiteralPath $chunk.FullName | Measure-Object -Line).Lines
            $complete = (Test-Path -LiteralPath $result) -and
                ((Get-Content -LiteralPath $result | Measure-Object -Line).Lines -eq $expected)
            if ($complete) {
                Write-Output "Skipping completed $($chunk.Name)"
                continue
            }
            Write-Output "Grading $($chunk.Name) ($expected questions)"
            $log = "$($chunk.FullName).log"
            & $pythonPath $evaluatorPath gpt-4o $chunk.FullName $datasetPath *> $log
            if ($LASTEXITCODE -ne 0) {
                throw "Official evaluator failed on $($chunk.Name); inspect $log"
            }
            $actual = (Get-Content -LiteralPath $result | Measure-Object -Line).Lines
            if ($actual -ne $expected) {
                throw "Official evaluator returned $actual of $expected labels for $($chunk.Name)"
            }
        }
    }
    & $projectPython -m evals.official_gpt4o summarize `
        --dataset $datasetPath --answers $answersPath --evaluator $evaluatorPath --output $outputPath
    if ($LASTEXITCODE -ne 0) { throw 'Independent summary validation failed' }
}
finally {
    if ($null -eq $previousOpenAIKey) {
        Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
    }
    else {
        $env:OPENAI_API_KEY = $previousOpenAIKey
    }
    if ($null -eq $previousPythonUtf8) {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONUTF8 = $previousPythonUtf8
    }
}

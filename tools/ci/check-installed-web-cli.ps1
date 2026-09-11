param([string]$OutDir = "target/ci/installed-web-cli")

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$evidenceDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutDir))
New-Item -ItemType Directory -Force -Path $evidenceDir | Out-Null
$runDir = Join-Path $evidenceDir ([Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $runDir | Out-Null
$temporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
$workDir = Join-Path $temporaryRoot ("bloom-installed-cli-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $workDir | Out-Null
$records = [System.Collections.Generic.List[object]]::new()
$report = [ordered]@{
    schema = "bloom-installed-web-cli-v1"
    status = "running"
    commands = $records
    runtime_rendering_verified = $false
}
$reportPath = Join-Path $evidenceDir "result.json"

function Save-Report {
    [System.IO.File]::WriteAllText($reportPath, ($report | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
}

function Invoke-Checked([string]$Tool, [string[]]$ToolArguments, [string]$Name) {
    $logPath = Join-Path $runDir "$Name.log"
    $errorLogPath = Join-Path $runDir "$Name.stderr.log"
    $started = Get-Date
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell wraps native stderr in ErrorRecords. Preserve the
        # tool's actual exit status and keep stderr out of npm's JSON stdout.
        $ErrorActionPreference = "Continue"
        $output = & $Tool @ToolArguments 2> $errorLogPath | Out-String
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    $output | Set-Content -Encoding utf8 $logPath
    $records.Add([ordered]@{
        name = $Name
        command = @($Tool) + $ToolArguments
        cwd = (Get-Location).Path
        exit_code = $code
        duration_seconds = [Math]::Round(((Get-Date) - $started).TotalSeconds, 3)
        log = $logPath
        stderr_log = $errorLogPath
    })
    Save-Report
    if ($code -ne 0) { throw "$Name failed with exit $code; see $logPath" }
    return $output
}

Save-Report
Push-Location $repoRoot
try {
    $npmCommand = (Get-Command npm -ErrorAction Stop).Source
    $nodeCommand = (Get-Command node -ErrorAction Stop).Source
    Invoke-Checked $nodeCommand @("--test", "tools/ci/test_web_build.cjs") "regressions" | Out-Null
    $packedText = Invoke-Checked $npmCommand @("pack", "--json", "--ignore-scripts", "--pack-destination", $workDir) "pack"
    $packed = @($packedText | ConvertFrom-Json)[0]
    $archive = Join-Path $workDir $packed.filename
    $report.archive_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    $project = Join-Path $workDir "clean project with spaces"
    New-Item -ItemType Directory -Path $project | Out-Null
    $manifest = @{ name = "bloom-installed-cli-smoke"; version = "1.0.0"; private = $true } | ConvertTo-Json
    [System.IO.File]::WriteAllText((Join-Path $project "package.json"), $manifest, [System.Text.UTF8Encoding]::new($false))
    Push-Location $project
    try {
        Invoke-Checked $npmCommand @("install", "--ignore-scripts", "--no-audit", "--no-fund", $archive) "install" | Out-Null
        $help = Invoke-Checked $npmCommand @("exec", "--", "bloom-web", "--help") "installed-help"
        if (-not $help.Contains("--output")) { throw "installed command did not print its usage" }
    } finally { Pop-Location }
    $report.status = "pass"
    Write-Output "PASS: packed and installed bloom-web command runs on Windows; rendering is a separate check."
} catch {
    $report.status = "fail"
    $report.error = $_.Exception.Message
    throw
} finally {
    Pop-Location
    Save-Report
    # Dependency packages stay out of CI evidence. Remove only this invocation's
    # checked temporary directory, never a computed repository/output ancestor.
    $resolvedWork = (Resolve-Path -LiteralPath $workDir).Path
    if ([System.IO.Path]::GetDirectoryName($resolvedWork) -ne $temporaryRoot -or
        ((Get-Item -LiteralPath $resolvedWork).Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "refusing cleanup outside the owned temporary directory: $resolvedWork"
    }
    Remove-Item -LiteralPath $resolvedWork -Recurse -Force
}

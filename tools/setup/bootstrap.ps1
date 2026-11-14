# アプリの起動に最低限必要なものを準備する。そろっていれば何もしない。start.cmd から呼ばれる。
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Refresh-Path {
    $extra = @("$env:USERPROFILE\.local\bin", "$env:LOCALAPPDATA\Microsoft\WinGet\Links", "$env:ProgramFiles\Git\cmd", "$env:ProgramFiles\nodejs")
    $saved = @([Environment]::GetEnvironmentVariable("Path", "Machine"), [Environment]::GetEnvironmentVariable("Path", "User"))
    $env:Path = (($saved + $extra) | Where-Object { $_ }) -join ";"
}

# Messages are Japanese on a Japanese Windows and English elsewhere.
$japanese = (Get-UICulture).Name -like "ja*"
function Say($ja, $en) { if ($japanese) { $ja } else { $en } }

function Need-Tool($command, $label, $package) {
    if (Get-Command $command -ErrorAction SilentlyContinue) { return }
    Refresh-Path
    if (Get-Command $command -ErrorAction SilentlyContinue) { return }
    Write-Host (Say "$label が見つかりません。" "$label was not found.")
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw (Say "winget がないため導入できません。$label を手動で入れてください。" "winget is not available, so it cannot be installed. Please install $label by hand.") }
    $answer = Read-Host (Say "winget で $package を導入しますか？ [Y/n]" "Install $package with winget? [Y/n]")
    if ($answer -match "^[nN]") { throw (Say "$label が必要です。" "$label is required.") }
    winget install --id $package -e --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw (Say "$label の導入に失敗しました。" "Installing $label failed.") }
    Refresh-Path
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw (Say "$label を導入しましたが見つかりません。PC を再起動してからもう一度お試しください。" "$label was installed but cannot be found. Restart the PC and try again.") }
}

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw (Say "$what に失敗しました。" "Failed: $what.") }
}

$python = Join-Path $root "backend\.venv\Scripts\python.exe"
$electron = Join-Path $root "electron\node_modules\.bin\electron.cmd"
$modules = Join-Path $root "frontend\node_modules"
$page = Join-Path $root "frontend\dist\index.html"
$backendReady = Join-Path $root "backend\.venv\.dsb-ready"
$lock = Join-Path $root "frontend\package-lock.json"
$installed = Join-Path $modules ".package-lock.json"

# 画面は、無いときと、元になるファイルより古いとき（更新のあと）に作り直す。
$pageStale = -not (Test-Path $page)
if (-not $pageStale) {
    $built = (Get-Item $page).LastWriteTimeUtc
    $publicDir = Join-Path $root "frontend\public"
    $publicFiles = if (Test-Path $publicDir) { @(Get-ChildItem $publicDir -Recurse -File) } else { @() }
    $sources = @(Get-ChildItem (Join-Path $root "frontend\src") -Recurse -File) + $publicFiles + @(Get-Item (Join-Path $root "frontend\index.html"), $lock)
    $pageStale = [bool]($sources | Where-Object { $_.LastWriteTimeUtc -gt $built } | Select-Object -First 1)
}
$modulesStale = -not (Test-Path $installed) -or ((Get-Item $lock).LastWriteTimeUtc -gt (Get-Item $installed).LastWriteTimeUtc)

$reqFile = Join-Path $root "backend\requirements.txt"
$backendStale = -not (Test-Path $backendReady) -or ((Get-Item $reqFile).LastWriteTimeUtc -gt (Get-Item $backendReady).LastWriteTimeUtc)

try {
    if ($backendStale) {
        Need-Tool "uv" "uv" "astral-sh.uv"
        Write-Host (Say "Python 環境を準備しています..." "Preparing the Python environment...")
        # An environment that exists is only updated. Creating it again fails while one of its processes runs
        # (the MCP server of an AI agent keeps python.exe open).
        if (-not (Test-Path $python)) {
            & uv venv --allow-existing --python 3.11 (Join-Path $root "backend\.venv")
            Check (Say "Python 環境の作成" "creating the Python environment")
        }
        & uv pip install --python $python -r $reqFile
        Check (Say "Python パッケージの導入" "installing the Python packages")
        New-Item -ItemType File -Force $backendReady | Out-Null
    }
    if (-not (Test-Path $electron) -or $pageStale) { Need-Tool "npm" "Node.js" "OpenJS.NodeJS.LTS" }
    if (-not (Test-Path $electron)) {
        Write-Host (Say "画面の実行環境を準備しています..." "Preparing the desktop shell...")
        Push-Location (Join-Path $root "electron"); & npm ci --no-fund --no-audit; $code = $LASTEXITCODE; Pop-Location
        $global:LASTEXITCODE = $code; Check (Say "Electron の導入" "installing Electron")
    }
    if ($pageStale) {
        Write-Host (Say "画面を作成しています..." "Building the page...")
        Push-Location (Join-Path $root "frontend")
        if ($modulesStale) { & npm ci --no-fund --no-audit; if ($LASTEXITCODE -ne 0) { Pop-Location; throw (Say "画面の依存パッケージの導入に失敗しました。" "Installing the page's packages failed.") } }
        & npm run build; $code = $LASTEXITCODE; Pop-Location
        $global:LASTEXITCODE = $code; Check (Say "画面の作成" "building the page")
    }
} catch {
    Write-Host ""
    Write-Host (Say "準備に失敗しました: $($_.Exception.Message)" "Setup failed: $($_.Exception.Message)") -ForegroundColor Red
    exit 1
}
exit 0

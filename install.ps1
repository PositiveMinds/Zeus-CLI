# Zeus installer for Windows (PowerShell)
# Usage:
#   irm https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI/main/install.ps1 | iex
#
# Downloads the latest zeus release, installs it under %LOCALAPPDATA%\zeus,
# and adds it to the user PATH.

$ErrorActionPreference = "Stop"

$RepoOwner = "PositiveMinds"
$RepoName = "Zeus-CLI"
$InstallDir = Join-Path $env:LOCALAPPDATA "zeus"
$BinFile = Join-Path $InstallDir "zeus.exe"

# Map the machine architecture to a release target name.
$target = "x86_64-pc-windows-msvc"

function Invoke-ZeusApi([string]$Path) {
    $headers = @{ "Accept" = "application/vnd.github+json"; "X-GitHub-Api-Version" = "2022-11-28" }
    if ($env:GITHUB_TOKEN) { $headers["Authorization"] = "Bearer $env:GITHUB_TOKEN" }
    return Invoke-RestMethod -Uri "https://api.github.com$Path" -Headers $headers
}

Write-Host "Installing zeus for target: $target" -ForegroundColor Cyan

# Resolve latest release; allow an explicit pinned tag via $env:ZEUS_VERSION.
$version = $env:ZEUS_VERSION
$asset = $null
if ($version) {
    $release = Invoke-ZeusApi "/repos/$RepoOwner/$RepoName/releases/tags/v$version"
    Write-Host "Installing zeus v$version from GitHub Release" -ForegroundColor Cyan
} else {
    $release = Invoke-ZeusApi "/repos/$RepoOwner/$RepoName/releases/latest"
    Write-Host "Installing latest zeus ($($release.tag_name))" -ForegroundColor Cyan
}

$asset = $release.assets | Where-Object { $_.name -like "zeus-$target.zip" }
if (-not $asset) {
    throw "No prebuilt binary found for target '$target' in release '$($release.tag_name)'. " +
          "Install from source instead: cargo install --git https://github.com/$RepoOwner/$RepoName"
}

$zip = Join-Path $env:TEMP "zeus-$target.zip"
Write-Host "Downloading $($asset.browser_download_url)" -ForegroundColor Cyan
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
$staging = Join-Path $env:TEMP ("zeus-extract-" + [guid]::NewGuid().ToString("N"))
Expand-Archive -Path $zip -DestinationPath $staging -Force
Copy-Item (Join-Path $staging "zeus.exe") -Destination $BinFile -Force
Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $zip -Force -ErrorAction SilentlyContinue

# Add to user PATH if not already present.
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    Write-Host "Added $InstallDir to your user PATH." -ForegroundColor Yellow
    $env:Path = "$env:Path;$InstallDir"   # also apply to this session
}

Write-Host ""
Write-Host "Installed zeus." -ForegroundColor Green
Write-Host "Open a new terminal and run:"
Write-Host "  zeus init" -ForegroundColor Cyan
Write-Host "  zeus doctor" -ForegroundColor Cyan
Write-Host "  zeus chat 'hello' --provider mock" -ForegroundColor Cyan
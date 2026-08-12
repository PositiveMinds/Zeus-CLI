Prebuilt binaries for Zeus __TAG__.

## Install

**npm** (any platform, if you have Node):
```bash
npm install -g zeus-code
```

**PowerShell** (Windows 10/11):
```powershell
irm https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.ps1 | iex
```

**cmd** (Windows):
```batch
curl -L https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.bat | cmd
```

**macOS / Linux**:
```bash
curl -fsSL https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.sh | sh
```

Pin this exact version with `ZEUS_VERSION=__VERSION__` (or `$env:ZEUS_VERSION` on PowerShell) before running a script installer.

## Getting started

```bash
zeus
```
Launches the interactive TUI. With no provider configured yet, it opens straight into setup — `/provider` picks one and walks you through pasting a key (or connecting a local runner like Ollama/LM Studio).

Other useful commands:
```bash
zeus doctor   # check every configured provider's readiness (key present, or reachable for local runners)
zeus init     # set up this project's .agent/ state (optional — run inside a repo)
zeus --help   # see every command
```

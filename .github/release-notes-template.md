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

## Troubleshooting

**`zeus` (or `zeus-code`) not recognized after `npm install -g zeus-code`**

`zeus-code` is just the npm package name — the command it installs is `zeus`, not `zeus-code`. If `zeus` still isn't found:

1. Confirm it's actually installed (not just claimed to be):
   ```bash
   npm ls -g zeus-code
   ```
   If this prints `(empty)` or doesn't list it at all, the install didn't really take — npm can report a false "up to date" against nothing installed. Force a clean reinstall:
   ```bash
   npm uninstall -g zeus-code
   npm cache clean --force
   npm install -g zeus-code
   ```

2. If it genuinely *is* installed but `zeus` still isn't found, npm's global bin folder likely isn't on `PATH`. Find it with:
   ```bash
   npm config get prefix
   ```
   **Windows (cmd)**: `echo %PATH% | findstr /i npm`
   **Windows (PowerShell)**: `$env:PATH -split ';' | Select-String npm`
   **macOS/Linux**: `echo $PATH | grep npm`

   If it's missing: on Windows, add the prefix folder via *System Properties → Environment Variables → User variables → Path → New*; on macOS/Linux, add `export PATH="$(npm config get prefix)/bin:$PATH"` to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

3. **Open a brand new terminal window** after any `PATH` change — an already-running session won't pick it up.

**`zeus doctor` shows a provider as unreachable/no key** — that's expected for anything you haven't configured yet. Only the providers you actually intend to use need a key (`export`/`$env:` the variable it names) or, for local runners (Ollama/LM Studio/llama.cpp), need to actually be running.

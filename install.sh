#!/bin/sh
# Zeus installer for macOS/Linux.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/PositiveMinds/Zeus-CLI-releases/main/install.sh | sh
#
# Downloads the latest zeus release, installs it under ~/.local/share/zeus/bin,
# and prints a PATH hint (this script can't durably edit your shell's PATH
# for you the way the Windows installers edit the registry — most shell
# rc files already treat that directory as user-owned, so the hint is a
# one-time `echo` block rather than an automatic edit).
#
# Pin a specific version with:
#   ZEUS_VERSION=1.2.3 curl -fsSL .../install.sh | sh

set -eu

REPO_OWNER="PositiveMinds"
REPO_NAME="Zeus-CLI-releases"
INSTALL_DIR="${ZEUS_INSTALL_DIR:-$HOME/.local/share/zeus/bin}"
BIN_FILE="$INSTALL_DIR/zeus"

os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
    Linux)
        case "$arch" in
            x86_64) target="x86_64-unknown-linux-gnu" ;;
            *) echo "Error: no prebuilt zeus binary for Linux/$arch yet." >&2
               echo "Install from source instead: cargo install --git https://github.com/$REPO_OWNER/$REPO_NAME" >&2
               exit 1 ;;
        esac
        archive_ext="tar.gz"
        ;;
    Darwin)
        case "$arch" in
            arm64)  target="aarch64-apple-darwin" ;;
            x86_64) target="x86_64-apple-darwin" ;;
            *) echo "Error: no prebuilt zeus binary for macOS/$arch yet." >&2
               exit 1 ;;
        esac
        archive_ext="tar.gz"
        ;;
    *)
        echo "Error: unsupported OS '$os' — this script covers Linux and macOS only." >&2
        echo "On Windows, use install.ps1 or install.bat instead." >&2
        exit 1
        ;;
esac

echo "Installing zeus for target: $target"

api_get() {
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        curl -fsSL -H "Accept: application/vnd.github+json" -H "Authorization: Bearer $GITHUB_TOKEN" "https://api.github.com$1"
    else
        curl -fsSL -H "Accept: application/vnd.github+json" "https://api.github.com$1"
    fi
}

if [ -n "${ZEUS_VERSION:-}" ]; then
    echo "Installing zeus v$ZEUS_VERSION from GitHub Release"
    release_json="$(api_get "/repos/$REPO_OWNER/$REPO_NAME/releases/tags/v$ZEUS_VERSION")" || {
        echo "Error: no release found for tag v$ZEUS_VERSION." >&2
        echo "Check https://github.com/$REPO_OWNER/$REPO_NAME/releases for available versions." >&2
        exit 1
    }
else
    release_json="$(api_get "/repos/$REPO_OWNER/$REPO_NAME/releases/latest")" || {
        echo "Error: couldn't reach the GitHub API, or no zeus release has been published yet." >&2
        echo "Check https://github.com/$REPO_OWNER/$REPO_NAME/releases for available versions." >&2
        exit 1
    }
    tag="$(printf '%s' "$release_json" | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')"
    echo "Installing latest zeus ($tag)"
fi

asset_name="zeus-$target.$archive_ext"
download_url="$(printf '%s' "$release_json" | grep -o "\"browser_download_url\": *\"[^\"]*$asset_name\"" | sed -E 's/.*"(https[^"]+)"/\1/')"
if [ -z "$download_url" ]; then
    echo "Error: no prebuilt binary named '$asset_name' found in that release." >&2
    echo "Install from source instead: cargo install --git https://github.com/$REPO_OWNER/$REPO_NAME" >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
archive_path="$tmp_dir/$asset_name"

echo "Downloading $download_url"
curl -fsSL -o "$archive_path" "$download_url"

mkdir -p "$INSTALL_DIR"
tar -xzf "$archive_path" -C "$tmp_dir"
found_bin="$(find "$tmp_dir" -type f -name zeus ! -name '*.tar.gz' | head -n1)"
if [ -z "$found_bin" ]; then
    echo "Error: archive didn't contain a 'zeus' binary." >&2
    exit 1
fi
cp "$found_bin" "$BIN_FILE"
chmod +x "$BIN_FILE"

echo ""
echo "Installed zeus to $BIN_FILE"
case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
        echo ""
        echo "Add it to your PATH (pick the line matching your shell, then restart your terminal):"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc   # bash"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc    # zsh"
        ;;
esac
echo ""
echo "Then run:"
echo "  zeus init"
echo "  zeus doctor"
echo "  zeus chat 'hello'"

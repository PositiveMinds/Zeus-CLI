#!/usr/bin/env node
"use strict";

// Resolves and execs the prebuilt zeus binary for the current platform —
// installed as one of this package's `optionalDependencies`, per the same
// pattern esbuild/swc/opencode use. This file has no other logic on
// purpose: the real CLI is the Rust binary, this is just the thing `npm
// install -g zeus-code` needs so a plain `zeus` command exists on PATH.

const { spawnSync } = require("child_process");

const PLATFORM_PACKAGES = {
  "win32-x64": "zeus-code-windows-x64",
  "linux-x64": "zeus-code-linux-x64",
  "darwin-x64": "zeus-code-darwin-x64",
  "darwin-arm64": "zeus-code-darwin-arm64",
};

function resolveBinary() {
  const key = `${process.platform}-${process.arch}`;
  const pkg = PLATFORM_PACKAGES[key];
  if (!pkg) {
    console.error(
      `zeus-code: no prebuilt binary for ${process.platform}/${process.arch}.\n` +
        `Supported: ${Object.keys(PLATFORM_PACKAGES).join(", ")}.\n` +
        "Build from source instead: https://github.com/PositiveMinds/Zeus-CLI"
    );
    process.exit(1);
  }
  const exe = process.platform === "win32" ? "zeus.exe" : "zeus";
  try {
    return require.resolve(`${pkg}/bin/${exe}`);
  } catch (err) {
    console.error(
      `zeus-code: the "${pkg}" optional dependency isn't installed.\n` +
        "npm sometimes skips optionalDependencies after a network hiccup or a\n" +
        "cache issue — try: npm install -g zeus-code --force"
    );
    process.exit(1);
  }
}

const bin = resolveBinary();
const result = spawnSync(bin, process.argv.slice(2), { stdio: "inherit" });
if (result.error) {
  console.error(`zeus-code: failed to launch zeus: ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status === null ? 1 : result.status);

#!/usr/bin/env bash
# Build every zeus scaffold whose toolchain is installed, proving each
# language/framework skeleton compiles (or at least parses). Runs in CI
# (see .github/workflows/scaffold-build.yml) and locally:
#   scripts/verify-scaffolds.sh ./target/release/zeus
set -uo pipefail

Z="${1:?usage: verify-scaffolds.sh <path-to-zeus-binary>}"
ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT

have() { command -v "$1" >/dev/null 2>&1; }

# scaffold <lang> <name> -- then run the build command(s) in the new dir.
run() {
  local lang="$1" name="$2"
  shift 2
  echo "== $lang =="
  "$Z" project --project-root "$ROOT" scaffold "$lang" "$name" >/dev/null \
    || { echo "scaffold $lang FAILED"; exit 1; }
  ( cd "$ROOT/$name" && "$@" ) || { echo "BUILD FAILED: $lang"; exit 1; }
}

py_syntax() {
  bash -c "set -e; for f in \$(find . -name '*.py'); do python3 -m py_compile \"\$f\"; done"
}

if have cargo; then
  run rust rust_app cargo build --quiet
fi

if have go; then
  run go go_app bash -c "go build ./..."
fi

if have javac && have mvn; then
  run java java_app mvn -q -B compile
fi

if have javac && have gradle; then
  run kotlin kt_app gradle -q --console=plain build
  run groovy gro_app gradle -q --console=plain build
fi

if have mvn; then
  run spring-boot srv_app mvn -q -B compile
fi

if have node && have npm; then
  run ts ts_app bash -c "npm install --no-audit --no-fund >/dev/null 2>&1 && npx tsc -p ."
  run js js_app node --check src/index.js
  run react rx_app bash -c "npm install --no-audit --no-fund >/dev/null 2>&1 && npm run build >/dev/null"
  run express exp_app bash -c "npm install --no-audit --no-fund >/dev/null 2>&1 && node -e \"require('./server.js');const http=require('http');http.get('http://127.0.0.1:3000/',r=>{let d='';r.on('data',c=>d+=c);r.on('end',()=>{console.log(r.statusCode===200?'express OK: '+d:'express FAIL');process.exit(r.statusCode===200?0:1)})}).on('error',e=>{console.error(e.message);process.exit(1)})\""
fi

if have dotnet; then
  run csharp cs_app bash -c "dotnet build -v q --nologo >/dev/null"
  run fsharp fs_app bash -c "dotnet build -v q --nologo >/dev/null"
  run vb vb_app bash -c "dotnet build -v q --nologo >/dev/null"
  run aspnet web_app bash -c "dotnet build -v q --nologo >/dev/null"
fi

if have python3; then
  run python py_app py_syntax
  run django dweb py_syntax
  run flask fweb py_syntax
fi

if have php; then
  run php php_app bash -c "php -l index.php >/dev/null"
  run laravel lar_app bash -c "php -l artisan >/dev/null && php -l routes/web.php >/dev/null"
fi

if have ruby; then
  run ruby rb_app bash -c "ruby -c main.rb >/dev/null"
  run rails rw_app bash -c "ruby -c config.ru >/dev/null && ruby -c app/controllers/application_controller.rb >/dev/null"
fi

echo "ALL SCAFFOLDS OK"
# npm distribution

This mirrors how esbuild/swc/opencode ship a Rust/Go binary through npm:
`zeus-code` is a thin wrapper whose only job is to exec the real binary,
which is installed for you as one of four platform-specific
`optionalDependencies` (npm only actually installs the one matching your
`os`/`cpu`, per each sub-package's `package.json`).

```
zeus-code/                   published as "zeus-code" — has the `zeus` bin entry
  bin/zeus.js                 resolves + spawns the right platform binary
  package.json

zeus-code-windows-x64/       published as "zeus-code-windows-x64"
zeus-code-linux-x64/         published as "zeus-code-linux-x64"
zeus-code-darwin-x64/        published as "zeus-code-darwin-x64"
zeus-code-darwin-arm64/      published as "zeus-code-darwin-arm64"
  bin/zeus(.exe)              <- NOT checked in; the release workflow copies
                                  the matching prebuilt binary here right
                                  before publishing (see
                                  `.github/workflows/release.yml`'s
                                  `publish-npm` job)
  package.json                 declares `os`/`cpu` so npm only installs
                                  this one on a matching machine
```

## Publishing

Publishing happens automatically from CI on every `v*` tag, once an
`NPM_TOKEN` repo secret (an npm automation token with publish rights) is
set. Until then the `publish-npm` job no-ops with a log line explaining why.

To publish by hand instead (e.g. the very first release, before CI is
trusted to do it):

```sh
# from a release's downloaded/extracted artifacts:
cp zeus-x86_64-pc-windows-msvc/zeus.exe npm/zeus-code-windows-x64/bin/
cp zeus-x86_64-unknown-linux-gnu/zeus   npm/zeus-code-linux-x64/bin/
cp zeus-x86_64-apple-darwin/zeus        npm/zeus-code-darwin-x64/bin/
cp zeus-aarch64-apple-darwin/zeus       npm/zeus-code-darwin-arm64/bin/
chmod +x npm/zeus-code-linux-x64/bin/zeus npm/zeus-code-darwin-x64/bin/zeus npm/zeus-code-darwin-arm64/bin/zeus

# bump the version in all 5 package.json files to match the release tag,
# keeping zeus-code's optionalDependencies versions in lockstep, then:
cd npm/zeus-code-windows-x64 && npm publish --access public && cd -
cd npm/zeus-code-linux-x64   && npm publish --access public && cd -
cd npm/zeus-code-darwin-x64  && npm publish --access public && cd -
cd npm/zeus-code-darwin-arm64 && npm publish --access public && cd -
# main package last — its optionalDependencies pin exact versions of the
# four packages above, so they need to already exist on the registry.
cd npm/zeus-code && npm publish --access public && cd -
```

#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/steve/modelsmc-reading-notes
STUDY="$REPO/artifacts/blind-filter-map-confirmation-v3-r4"
PROTOCOL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r4.json"
METHOD_SEAL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r4.method-seal.json"
PYTHON="$REPO/.venv/bin/python"
METHOD_SHA=a0dba72f8d2df04668fab9799d2a357bf47b389421854949d4fdc73c4056f9dd
CUSTODY_SHA=7f978f0e68d9b3034b651581ee67b563a8d651b28bc6aa01d39ccb45b3e27077
PROVIDER_SHA=b5e343f52202a6f0935fca96cfacda6b6fdf8ae3c961756e9f93459ae9884482

cd "$REPO"
grep -q '"event":"driver_completed"' "$STUDY/driver.log"
test "$(find "$STUDY/runs" -mindepth 3 -maxdepth 3 -type f -name result.json | wc -l)" -eq 24
test ! -e "$STUDY/analysis/analysis.json"
test ! -e "$STUDY/public-bundle-seal"

runuser -u nobody -- "$PYTHON" -m research.analyze_blind_filter_map_confirmation_v3 \
  --repo-root "$REPO" \
  --study-protocol "$PROTOCOL" \
  --method-seal "$METHOD_SEAL" \
  --expected-method-seal-sha256 "$METHOD_SHA" \
  --custody-seal "$STUDY/seals/custody-seal.json" \
  --expected-custody-seal-sha256 "$CUSTODY_SHA" \
  --provider-seal "$STUDY/seals/provider-seal.json" \
  --expected-provider-seal-sha256 "$PROVIDER_SHA" \
  --runs-root "$STUDY/runs" \
  --output "$STUDY/analysis/analysis.json"

runuser -u nobody -- "$PYTHON" -m research.seal_blind_filter_map_confirmation_v3_public \
  --runs-root "$STUDY/runs" \
  --analysis "$STUDY/analysis/analysis.json" \
  --output "$STUDY/public-bundle-seal" \
  --preflight-only

runuser -u nobody -- "$PYTHON" -m research.seal_blind_filter_map_confirmation_v3_public \
  --runs-root "$STUDY/runs" \
  --analysis "$STUDY/analysis/analysis.json" \
  --output "$STUDY/public-bundle-seal"

printf 'analysis_sha256='
sha256sum "$STUDY/analysis/analysis.json" | cut -d' ' -f1
printf 'public_bundle_sha256='
tr -d '\n' < "$STUDY/public-bundle-seal/BUNDLE_SHA256"
printf '\n'

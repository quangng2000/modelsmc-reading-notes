#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/steve/modelsmc-reading-notes
STUDY="$REPO/artifacts/blind-filter-map-confirmation-v3-r5"
PROTOCOL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r5.json"
METHOD_SEAL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r5.method-seal.json"
PYTHON="$REPO/.venv/bin/python"
METHOD_SHA=5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228
CUSTODY_SHA=5980609510079cb85ce54f2bc34c75c265ee8a9e26afdc1a368accce29a39427
PROVIDER_SHA=22059898fedef2284f97f786680972a78e11408dc226c44b9603db0c8a2b527d
PRIVATE_REVEAL=/root/blind-filter-map-confirmation-v3-r5-custody/generated/suite/private/reveal.json
DRIVER_PID=${1:?driver PID is required}

record_exit() {
  local status=$?
  if ((status != 0)); then
    printf '{"event":"postrun_failed","exit_code":%d,"utc":"%s"}\n' \
      "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  printf '%d\n' "$status" > "$STUDY/postrun.exit"
}
trap record_exit EXIT

cd "$REPO"
test ! -e "$STUDY/postrun.exit"
if runuser -u nobody -- test -r "$PRIVATE_REVEAL"; then
  printf '{"event":"private_boundary_failed"}\n'
  exit 51
fi

printf '{"event":"postrun_watcher_started","driver_pid":%d,"utc":"%s"}\n' \
  "$DRIVER_PID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while [[ ! -f "$STUDY/driver.exit" ]]; do
  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    printf '{"event":"driver_disappeared_without_status","driver_pid":%d}\n' "$DRIVER_PID"
    exit 52
  fi
  sleep 10
done

driver_status=$(tr -d '[:space:]' < "$STUDY/driver.exit")
if [[ "$driver_status" != 0 ]]; then
  printf '{"event":"analysis_skipped_after_driver_failure","driver_exit":%s}\n' \
    "$driver_status"
  exit 53
fi
grep -q '"event":"driver_completed"' "$STUDY/driver.log"
test "$(find "$STUDY/runs" -mindepth 3 -maxdepth 3 -type f -name result.json | wc -l)" -eq 24
test ! -e "$STUDY/analysis/analysis.json"
test ! -e "$STUDY/public-bundle-seal"
if runuser -u nobody -- test -r "$PRIVATE_REVEAL"; then
  printf '{"event":"private_boundary_failed_before_analysis"}\n'
  exit 54
fi

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

printf '{"event":"postrun_completed","analysis_sha256":"%s","public_bundle_sha256":"%s","utc":"%s"}\n' \
  "$(sha256sum "$STUDY/analysis/analysis.json" | cut -d' ' -f1)" \
  "$(tr -d '\n' < "$STUDY/public-bundle-seal/BUNDLE_SHA256")" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/steve/modelsmc-reading-notes
STUDY="$REPO/artifacts/blind-filter-map-confirmation-v3-r5"
PROTOCOL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r5.json"
METHOD_SEAL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r5.method-seal.json"
PYTHON="$REPO/.venv/bin/python"
BASE_URL=http://127.0.0.1:18000/v1
METHOD_SHA=5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228
CUSTODY_SHA=5980609510079cb85ce54f2bc34c75c265ee8a9e26afdc1a368accce29a39427
PROVIDER_SHA=22059898fedef2284f97f786680972a78e11408dc226c44b9603db0c8a2b527d
PRIVATE_REVEAL=/root/blind-filter-map-confirmation-v3-r5-custody/generated/suite/private/reveal.json

record_exit() {
  local status=$?
  if ((status != 0)); then
    printf '{"event":"driver_failed","exit_code":%d,"utc":"%s"}\n' \
      "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  printf '%d\n' "$status" > "$STUDY/driver.exit"
}
trap record_exit EXIT

cd "$REPO"
test ! -e "$STUDY/driver.exit"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = \
  36bd14082c767f47327b73a1ee57f44763fcd97b2bea6dcca0637a8cddc83d2d
test "$(sha256sum "$METHOD_SEAL" | cut -d' ' -f1)" = "$METHOD_SHA"
test "$(sha256sum "$STUDY/seals/custody-seal.json" | cut -d' ' -f1)" = "$CUSTODY_SHA"
test "$(sha256sum "$STUDY/seals/provider-seal.json" | cut -d' ' -f1)" = "$PROVIDER_SHA"
if runuser -u nobody -- test -r "$PRIVATE_REVEAL"; then
  printf '{"event":"private_boundary_failed"}\n'
  exit 41
fi
test -z "$(find "$STUDY/runs" -mindepth 1 -print -quit)"

printf '{"event":"driver_started","utc":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for index in 01 02 03 04 05 06 07 08 09 10 11 12; do
  for arm in grammar-random llm; do
    task_id="blind-v3-$index"
    printf '{"event":"arm_preflight_started","task_id":"%s","arm":"%s","utc":"%s"}\n' \
      "$task_id" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    runuser -u nobody -- "$PYTHON" -m research.run_blind_filter_map_confirmation_v3 \
      --repo-root "$REPO" \
      --study-protocol "$PROTOCOL" \
      --method-seal "$METHOD_SEAL" \
      --expected-method-seal-sha256 "$METHOD_SHA" \
      --custody-seal "$STUDY/seals/custody-seal.json" \
      --expected-custody-seal-sha256 "$CUSTODY_SHA" \
      --provider-seal "$STUDY/seals/provider-seal.json" \
      --expected-provider-seal-sha256 "$PROVIDER_SHA" \
      --task-id "$task_id" \
      --arm "$arm" \
      --runs-root "$STUDY/runs" \
      --base-url "$BASE_URL" \
      --python-executable "$PYTHON" \
      --preflight-only
    printf '{"event":"arm_preflight_completed","task_id":"%s","arm":"%s","utc":"%s"}\n' \
      "$task_id" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"event":"arm_started","task_id":"%s","arm":"%s","utc":"%s"}\n' \
      "$task_id" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    runuser -u nobody -- "$PYTHON" -m research.run_blind_filter_map_confirmation_v3 \
      --repo-root "$REPO" \
      --study-protocol "$PROTOCOL" \
      --method-seal "$METHOD_SEAL" \
      --expected-method-seal-sha256 "$METHOD_SHA" \
      --custody-seal "$STUDY/seals/custody-seal.json" \
      --expected-custody-seal-sha256 "$CUSTODY_SHA" \
      --provider-seal "$STUDY/seals/provider-seal.json" \
      --expected-provider-seal-sha256 "$PROVIDER_SHA" \
      --task-id "$task_id" \
      --arm "$arm" \
      --runs-root "$STUDY/runs" \
      --base-url "$BASE_URL" \
      --python-executable "$PYTHON"
    printf '{"event":"arm_completed","task_id":"%s","arm":"%s","utc":"%s"}\n' \
      "$task_id" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
done

result_count=$(find "$STUDY/runs" -mindepth 3 -maxdepth 3 -type f -name result.json | wc -l)
if [[ "$result_count" -ne 24 ]]; then
  printf '{"event":"result_count_failed","result_count":%d}\n' "$result_count"
  exit 42
fi
printf '{"event":"driver_completed","result_count":%d,"utc":"%s"}\n' \
  "$result_count" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/steve/modelsmc-reading-notes
STUDY="$REPO/artifacts/blind-filter-map-confirmation-v3-r3"
PROTOCOL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r3.json"
METHOD_SEAL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r3.method-seal.json"
PYTHON="$REPO/.venv/bin/python"
BASE_URL=http://127.0.0.1:18000/v1
METHOD_SHA=6de10073ab209f286fe6ea89e203cd970ca3e5e1fef1cd1186b7ae5f2abec6a3
CUSTODY_SHA=5cadffe08703607a3e2a6b73e08ffc0ad50f59e3e2b63042ba692a878fbc1b15
PROVIDER_SHA=1dd1baab5616d9ef4c22bea4056f582955ab7a8b92e64e0e5c5c29910b11d58f
PRIVATE_REVEAL=/root/blind-filter-map-confirmation-v3-r3-custody/generated/suite/private/reveal.json

failure_report() {
  local status=$?
  if ((status != 0)); then
    printf '{"event":"driver_failed","exit_code":%d,"utc":"%s"}\n' \
      "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
}
trap failure_report EXIT

cd "$REPO"
if runuser -u nobody -- test -r "$PRIVATE_REVEAL"; then
  printf '{"event":"private_boundary_failed"}\n'
  exit 41
fi

printf '{"event":"driver_started","utc":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for index in 01 02 03 04 05 06 07 08 09 10 11 12; do
  for arm in grammar-random llm; do
    task_id="blind-v3-$index"
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

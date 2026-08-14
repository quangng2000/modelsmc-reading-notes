# Blind filter-map confirmation v3, clean successor r5 runbook

Revision r3 is an immutable aborted pre-provider attempt. Its first
`grammar-random` arm failed frozen-invocation validation before any program
execution or provider call because the derived run record omitted `epsilon`
and `evidence_scale`. Never resume, repair, or reuse the r3 study, custody
directory, secret, or task suite.

Revision r4 is an immutable post-analysis, pre-unblinding
infrastructure/method-integrity abort. All 24 arms and reveal-free analysis
completed, but its frozen public sealer rejected the legitimate
`round-01.json` through `round-04.json` ledgers. Its diagnostic analysis is
not a confirmatory outcome. Never resume, repair, unblind, or reuse the r4
study, custody directory, secret, or task suite.

The corrected r5 sealer was frozen before any r5 secret preparation, then
exercised against the immutable r4 public runs and analysis only. The
descriptive posthoc evidence seal is outside the r4 tree at
`artifacts/blind-filter-map-confirmation-v3-r4-posthoc-abort-evidence-r5/`.
It inventories 385 reveal-free files with bundle SHA-256
`84ef009c3b64aa79d9bd7df054c3685b7585c1166872ea71fb67539b29e3ee6b`.
The r4 source-tree digest was identical before and after that read-only pass:
`4459c1bba65a6afdfc0f9fdbd60fd75e88a857f18d857bb7e5798aa015e2383e`.
This evidence does not rescue r4 and no r4 reveal was read.

The immutable r5 method is:

- `research/protocol-blind-filter-map-confirmation-v3-r5.json`
  - SHA-256 `36bd14082c767f47327b73a1ee57f44763fcd97b2bea6dcca0637a8cddc83d2d`
- `research/protocol-blind-filter-map-confirmation-v3-r5.method-seal.json`
  - SHA-256 `5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228`

The r5 runner SHA-256 is
`f9399dbe009a290ada718e9f1789d543bd4027342f3c56a5daa2a3f15e660a0c`.
It is byte-identical to r4. The common code, generator, harness, analysis,
dependency bundle, ModelSMC Python tree, prompt binding, run seeds, task
distribution, search protocol, and every scientific protocol field are also
identical to r4. The only execution-source binding changed from r4 is the
public sealer, from
`851f22c7dff76c9aafe1a6961d661e0475466e942ca215f0c119151a9f516413`
to `c1003027cd853e53e3c3a40591d79f4d3dc00468956e81e8c24785b9cfff0220`;
its allowlist now requires exactly four round ledgers. The only changed test
source is its focused sealer test, SHA-256
`195593d0db0a0837ae130f2b2bbba39a0c4bcce64b168f10915ac285285583a2`,
which tests the actual four-round layout and rejects missing or extra rounds.

Before this runbook was released, the complete bound test bundle passed (35
tests), every one of the 24 task-arm derived invocations passed the underlying
harness validator in a focused frozen test, and a static audit found no
harness-required per-run key missing from the launcher. That test intentionally
bypasses only the launcher's serial predecessor check so all 24 records can be
validated before a run tree exists. Operationally, sealed launcher preflights
must be just in time because each later arm requires its frozen predecessor's
result. The frozen method also validated as `nobody`, and the frozen sealer
passed on the actual 24-arm r4 public layout. No r5 secret, suite, provider
seal, or completion call existed at freeze time.

## Fixed r5 paths

Run every command below, in order, in one root Bash session. Establish the
fail-closed shell before evaluating any path, hash, preparation, or commitment
gate:

```sh
set -euo pipefail
REPO=/workspace/steve/modelsmc-reading-notes
CUSTODY=/root/blind-filter-map-confirmation-v3-r5-custody
STUDY="$REPO/artifacts/blind-filter-map-confirmation-v3-r5"
PROTOCOL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r5.json"
METHOD_SEAL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r5.method-seal.json"
METHOD_SHA=5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228
PROTOCOL_SHA=36bd14082c767f47327b73a1ee57f44763fcd97b2bea6dcca0637a8cddc83d2d
PYTHON="$REPO/.venv/bin/python"
BASE_URL=http://127.0.0.1:18000/v1
```

Fail closed if either new r5 path already exists. Do not delete or repurpose an
existing path:

```sh
test ! -e "$CUSTODY"
test ! -e "$STUDY"
cd "$REPO"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "$PROTOCOL_SHA"
test "$(sha256sum "$METHOD_SEAL" | cut -d' ' -f1)" = "$METHOD_SHA"
```

## 1. Create a new r5 secret after the r5 freeze

```sh
install -d -m 700 "$CUSTODY"
cd "$REPO"
"$PYTHON" -m research.generate_blind_filter_map_confirmation_v3 prepare-secret \
  --repo-root "$REPO" \
  --study-protocol "$PROTOCOL" \
  --method-seal "$METHOD_SEAL" \
  --expected-method-seal-sha256 "$METHOD_SHA" \
  --output "$CUSTODY/suite-secret.bin" \
  | tee "$CUSTODY/prepare-secret-public-output.json"
stat -c '%a %s %n' "$CUSTODY/suite-secret.bin"
R5_COMMITMENT=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["task_secret_commitment_sha256"])' \
  "$CUSTODY/prepare-secret-public-output.json")
test "$R5_COMMITMENT" != \
  10e3e473540d07378c37f833a9a8faf6e8de8cd0a3c0083fa53d05f2340f6a61
test "$R5_COMMITMENT" != \
  a3440d0d8ec1c88e7d57870966a14fa40a7b4cf165ba809cadc2e0386892e8b7
printf '%s\n' "$R5_COMMITMENT"
```

The `stat` command must report mode `600` and size `32`. Only the commitment
may leave custody. The explicit inequality checks prevent accidental reuse of
the r3 or r4 secret; if either fails, stop without sealing or generating.

## 2. Seal custody before generating tasks

```sh
cd "$REPO"
"$PYTHON" -m research.generate_blind_filter_map_confirmation_v3 seal-custody \
  --repo-root "$REPO" \
  --study-protocol "$PROTOCOL" \
  --method-seal "$METHOD_SEAL" \
  --expected-method-seal-sha256 "$METHOD_SHA" \
  --secret-file "$CUSTODY/suite-secret.bin" \
  --output "$CUSTODY/custody-seal.json"
CUSTODY_SHA=$(sha256sum "$CUSTODY/custody-seal.json" | cut -d' ' -f1)
printf '%s\n' "$CUSTODY_SHA"
```

Record `CUSTODY_SHA` externally before generation.

## 3. Generate the new r5 suite exactly once

```sh
install -d -m 700 "$CUSTODY/generated"
cd "$REPO"
"$PYTHON" -m research.generate_blind_filter_map_confirmation_v3 generate \
  --repo-root "$REPO" \
  --study-protocol "$PROTOCOL" \
  --method-seal "$METHOD_SEAL" \
  --expected-method-seal-sha256 "$METHOD_SHA" \
  --secret-file "$CUSTODY/suite-secret.bin" \
  --custody-seal "$CUSTODY/custody-seal.json" \
  --expected-custody-seal-sha256 "$CUSTODY_SHA" \
  --output "$CUSTODY/generated/suite"
```

Create the new public study boundary without copying private data:

```sh
install -d -o root -g nogroup -m 1775 "$STUDY"
install -d -o root -g root -m 755 "$STUDY/seals" "$STUDY/suite"
install -d -o nobody -g nogroup -m 755 "$STUDY/runs" "$STUDY/analysis"
install -m 644 "$CUSTODY/custody-seal.json" "$STUDY/seals/custody-seal.json"
cp -a "$CUSTODY/generated/suite/public" "$STUDY/suite/public"
test ! -e "$STUDY/suite/private"
```

The r3 and r4 public suites must not be copied or referenced.

## 4. Seal endpoint health and provider boundary before completions

Create `$STUDY/seals/endpoint-health.json` using only health/model inspection.
It must bind endpoint `http://127.0.0.1:18000/v1`, model `gpt-oss-120b`, and
revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. Do not issue a completion.

```sh
HIDDEN_SHA=$(sha256sum "$CUSTODY/generated/suite/private/reveal.json" | cut -d' ' -f1)
cd "$REPO"
"$PYTHON" -m research.generate_blind_filter_map_confirmation_v3 seal-provider \
  --repo-root "$REPO" \
  --study-protocol "$PROTOCOL" \
  --method-seal "$METHOD_SEAL" \
  --expected-method-seal-sha256 "$METHOD_SHA" \
  --custody-seal "$STUDY/seals/custody-seal.json" \
  --expected-custody-seal-sha256 "$CUSTODY_SHA" \
  --public-manifest "$STUDY/suite/public/manifest.json" \
  --hidden-target-manifest-sha256 "$HIDDEN_SHA" \
  --endpoint-health-record "$STUDY/seals/endpoint-health.json" \
  --output "$STUDY/seals/provider-seal.json"
PROVIDER_SHA=$(sha256sum "$STUDY/seals/provider-seal.json" | cut -d' ' -f1)
printf '%s\n' "$PROVIDER_SHA"
```

Externally record `PROVIDER_SHA`. From this point until the public bundle is
sealed, the runner and analyst must not be able to traverse the r5 custody
directory.

## 5. Isolated preflight

The venv interpreter resolves through `/root/.local`; `/root` may be mode
`0711`, but `$CUSTODY` and its private directory must remain `0700`, and the
reveal must remain `0600`.

```sh
chmod 711 /root
chmod 700 "$CUSTODY" "$CUSTODY/generated/suite/private"
chmod 600 "$CUSTODY/generated/suite/private/reveal.json"
runuser -u nobody -- test ! -r "$CUSTODY/generated/suite/private/reveal.json"
cd "$REPO"
runuser -u nobody -- "$PYTHON" -m research.run_blind_filter_map_confirmation_v3 \
  --repo-root "$REPO" \
  --study-protocol "$PROTOCOL" \
  --method-seal "$METHOD_SEAL" \
  --expected-method-seal-sha256 "$METHOD_SHA" \
  --custody-seal "$STUDY/seals/custody-seal.json" \
  --expected-custody-seal-sha256 "$CUSTODY_SHA" \
  --provider-seal "$STUDY/seals/provider-seal.json" \
  --expected-provider-seal-sha256 "$PROVIDER_SHA" \
  --task-id blind-v3-01 \
  --arm grammar-random \
  --runs-root "$STUDY/runs" \
  --base-url "$BASE_URL" \
  --python-executable "$PYTHON" \
  --preflight-only
test -z "$(find "$STUDY/runs" -mindepth 1 -print -quit)"
```

This real-bound empty-tree preflight must exit zero, issue no provider
completion, and leave the runs directory empty. Stop if it fails. Every arm is
then preflighted just in time in the serial execution loop; the first arm is
therefore deliberately validated again immediately before its one execution.

## 6. Execute exactly 24 arms

Run serially in task order, grammar-random then LLM. Stop on the first nonzero
exit. Never retry, resume, backfill, replace, skip, or parallelize an arm.

```sh
set -e
cd "$REPO"
for INDEX in 01 02 03 04 05 06 07 08 09 10 11 12; do
  for ARM in grammar-random llm; do
    runuser -u nobody -- "$PYTHON" -m research.run_blind_filter_map_confirmation_v3 \
      --repo-root "$REPO" \
      --study-protocol "$PROTOCOL" \
      --method-seal "$METHOD_SEAL" \
      --expected-method-seal-sha256 "$METHOD_SHA" \
      --custody-seal "$STUDY/seals/custody-seal.json" \
      --expected-custody-seal-sha256 "$CUSTODY_SHA" \
      --provider-seal "$STUDY/seals/provider-seal.json" \
      --expected-provider-seal-sha256 "$PROVIDER_SHA" \
      --task-id "blind-v3-$INDEX" \
      --arm "$ARM" \
      --runs-root "$STUDY/runs" \
      --base-url "$BASE_URL" \
      --python-executable "$PYTHON" \
      --preflight-only
    runuser -u nobody -- "$PYTHON" -m research.run_blind_filter_map_confirmation_v3 \
      --repo-root "$REPO" \
      --study-protocol "$PROTOCOL" \
      --method-seal "$METHOD_SEAL" \
      --expected-method-seal-sha256 "$METHOD_SHA" \
      --custody-seal "$STUDY/seals/custody-seal.json" \
      --expected-custody-seal-sha256 "$CUSTODY_SHA" \
      --provider-seal "$STUDY/seals/provider-seal.json" \
      --expected-provider-seal-sha256 "$PROVIDER_SHA" \
      --task-id "blind-v3-$INDEX" \
      --arm "$ARM" \
      --runs-root "$STUDY/runs" \
      --base-url "$BASE_URL" \
      --python-executable "$PYTHON"
  done
done
```

Each just-in-time preflight must exit zero and write nothing; the corresponding
arm is then executed exactly once. The predecessor results present for later
preflights are the immutable outputs of earlier arms in this same serial loop.

Any process failure outside the harness's totalized semantics aborts r5. Keep
all partial bytes and freeze a new revision; do not restart r5.

## 7. Reveal-free analysis and public seal

```sh
cd "$REPO"
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
```

Externally record `$STUDY/public-bundle-seal/BUNDLE_SHA256` before unblinding.

## 8. Unblind only after the public bundle seal

```sh
cd "$REPO"
"$PYTHON" -m research.generate_blind_filter_map_confirmation_v3 verify \
  --public "$STUDY/suite/public" \
  --reveal "$CUSTODY/generated/suite/private/reveal.json" \
  --custody-seal "$STUDY/seals/custody-seal.json"
```

Publish the r5 secret, reveal, rejection log, custody/provider seals, public
bundle inventory, reveal-free analysis, and the complete r3 and r4 abort
records.

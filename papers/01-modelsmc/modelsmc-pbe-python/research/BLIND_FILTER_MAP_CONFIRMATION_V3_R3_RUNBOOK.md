# Blind filter-map confirmation v3, Stage-1 r3 runbook

This runbook is operational guidance, not a method input. The immutable method is:

- `research/protocol-blind-filter-map-confirmation-v3-r3.json`
  - SHA-256 `f1c4624282de6ec526f8cfe6dce794e9b908c1d0300af1572e931cd5d7132b8f`
- `research/protocol-blind-filter-map-confirmation-v3-r3.method-seal.json`
  - SHA-256 `6de10073ab209f286fe6ea89e203cd970ca3e5e1fef1cd1186b7ae5f2abec6a3`

Revisions r1 and r2 were superseded before secret preparation. Never use them for
custody, generation, or a run.

## Fixed remote paths

```sh
REPO=/workspace/steve/modelsmc-reading-notes
CUSTODY=/root/blind-filter-map-confirmation-v3-r3-custody
STUDY="$REPO/artifacts/blind-filter-map-confirmation-v3-r3"
PROTOCOL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r3.json"
METHOD_SEAL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r3.method-seal.json"
METHOD_SHA=6de10073ab209f286fe6ea89e203cd970ca3e5e1fef1cd1186b7ae5f2abec6a3
PYTHON="$REPO/.venv/bin/python"
BASE_URL=http://127.0.0.1:18000/v1
```

Do not run this procedure until the current developmental scheduler has finished
and an operator has verified the copied r3 source tree against the method seal.

## 1. Custodian: prepare a new secret

The custodian directory must not be readable by the experiment runner.

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
```

Expected new private artifact: `$CUSTODY/suite-secret.bin`, raw 32 bytes, mode
0600. The command prints only `task_secret_commitment_sha256`.

## 2. Custodian: seal the commitment before generation

```sh
cd "$REPO"
"$PYTHON" -m research.generate_blind_filter_map_confirmation_v3 seal-custody \
  --repo-root "$REPO" \
  --study-protocol "$PROTOCOL" \
  --method-seal "$METHOD_SEAL" \
  --expected-method-seal-sha256 "$METHOD_SHA" \
  --secret-file "$CUSTODY/suite-secret.bin" \
  --output "$CUSTODY/custody-seal.json"
sha256sum "$CUSTODY/custody-seal.json"
```

Record the printed digest as `CUSTODY_SHA`. Copy only `custody-seal.json` into
`$STUDY/seals/custody-seal.json`; never copy the secret.

## 3. Custodian: generate exactly once

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

Expected private artifact: `$CUSTODY/generated/suite/private/reveal.json`.
Expected public boundary: `$CUSTODY/generated/suite/public/`. Copy the public
directory to `$STUDY/suite/public/`, preserving exact bytes. Do not copy the
private directory or the one-use secret receipt.

## 4. Operator: endpoint-health record

Before any completion call, perform only read-only health/model inspection and
write `$STUDY/seals/endpoint-health.json`. It must include at least:

```json
{
  "schema": "blind-filter-map-confirmation-v3-endpoint-health-v1",
  "endpoint": "http://127.0.0.1:18000/v1",
  "model": "gpt-oss-120b",
  "model_revision": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
}
```

It may also record `/version`, `/models`, vLLM arguments, time, and deployed
hashes. The generator checks exact `model` and `model_revision` fields.

## 5. Custodian/operator: pre-completion provider seal

Hash the private reveal without reading it into the runner process:

```sh
HIDDEN_SHA=$(sha256sum "$CUSTODY/generated/suite/private/reveal.json" | cut -d' ' -f1)
install -d "$STUDY/seals"
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
sha256sum "$STUDY/seals/provider-seal.json"
```

Record the digest as `PROVIDER_SHA`. From this point through the public bundle
seal, the runner must have no filesystem access to `$CUSTODY`.

## 6. Preflight, then run all matched arms

The frozen order is grammar-random then llm for each task, tasks 01 through 12.
Every command derives the exact 29-slot schedule from Stage 1. The
grammar-random arm makes zero provider calls.

For a nonmutating first preflight:

```sh
cd "$REPO"
"$PYTHON" -m research.run_blind_filter_map_confirmation_v3 \
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
```

Then run all 24 arms in the exact order. A shell loop may invoke the same command
without `--preflight-only`, but it must stop immediately on any nonzero exit and
must not retry, backfill, replace, or skip a run.

```sh
set -e
cd "$REPO"
for INDEX in 01 02 03 04 05 06 07 08 09 10 11 12; do
  for ARM in grammar-random llm; do
    "$PYTHON" -m research.run_blind_filter_map_confirmation_v3 \
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

## 7. Reveal-free analysis

```sh
install -d "$STUDY/analysis"
cd "$REPO"
"$PYTHON" -m research.analyze_blind_filter_map_confirmation_v3 \
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
```

Any analyzer exit 2 is an aborted confirmatory attempt. Do not unblind or rerun.

## 8. Seal exact public bytes before unblinding

```sh
cd "$REPO"
"$PYTHON" -m research.seal_blind_filter_map_confirmation_v3_public \
  --runs-root "$STUDY/runs" \
  --analysis "$STUDY/analysis/analysis.json" \
  --output "$STUDY/public-bundle-seal" \
  --preflight-only
"$PYTHON" -m research.seal_blind_filter_map_confirmation_v3_public \
  --runs-root "$STUDY/runs" \
  --analysis "$STUDY/analysis/analysis.json" \
  --output "$STUDY/public-bundle-seal"
```

Externally record `$STUDY/public-bundle-seal/BUNDLE_SHA256` before granting the
runner or analyst access to the reveal.

## 9. Unblind and verify

Only after Step 8 is externally recorded:

```sh
cd "$REPO"
"$PYTHON" -m research.generate_blind_filter_map_confirmation_v3 verify \
  --public "$STUDY/suite/public" \
  --reveal "$CUSTODY/generated/suite/private/reveal.json" \
  --custody-seal "$STUDY/seals/custody-seal.json"
```

Publish the secret, reveal, rejection logs, custody/provider seals, both
supersession records, exact public bundle inventory, and the reveal-free
analysis. Verification is not permitted to alter the already sealed analysis.

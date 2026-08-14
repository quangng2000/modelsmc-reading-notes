# Blind filter-map confirmation v3, clean successor r4 runbook

Revision r3 is an immutable aborted pre-provider attempt. Its first
`grammar-random` arm failed frozen-invocation validation before any program
execution or provider call because the derived run record omitted `epsilon`
and `evidence_scale`. Never resume, repair, or reuse the r3 study, custody
directory, secret, or task suite.

The immutable r4 method is:

- `research/protocol-blind-filter-map-confirmation-v3-r4.json`
  - SHA-256 `9c44d1391413163d93dd538288b470e63ea3aedb1616b77dd82f65fac4d95990`
- `research/protocol-blind-filter-map-confirmation-v3-r4.method-seal.json`
  - SHA-256 `a0dba72f8d2df04668fab9799d2a357bf47b389421854949d4fdc73c4056f9dd`

The r4 runner SHA-256 is
`f9399dbe009a290ada718e9f1789d543bd4027342f3c56a5daa2a3f15e660a0c`.
Removing exactly the two r4 run-record lines reconstructs the frozen r3 runner
SHA-256 `2e50da2050d9efe73c28397c7118c532426ea23ab36ffddd3cfeb997689b6f7d`.

Before this runbook was released, the complete bound test bundle passed (33
tests), every one of the 24 task-arm derived invocations passed the underlying
harness validator, and a static audit found no harness-required per-run key
missing from the launcher. No r4 secret, suite, provider seal, or completion
call existed at freeze time.

## Fixed r4 paths

```sh
REPO=/workspace/steve/modelsmc-reading-notes
CUSTODY=/root/blind-filter-map-confirmation-v3-r4-custody
STUDY="$REPO/artifacts/blind-filter-map-confirmation-v3-r4"
PROTOCOL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r4.json"
METHOD_SEAL="$REPO/research/protocol-blind-filter-map-confirmation-v3-r4.method-seal.json"
METHOD_SHA=a0dba72f8d2df04668fab9799d2a357bf47b389421854949d4fdc73c4056f9dd
PROTOCOL_SHA=9c44d1391413163d93dd538288b470e63ea3aedb1616b77dd82f65fac4d95990
PYTHON="$REPO/.venv/bin/python"
BASE_URL=http://127.0.0.1:18000/v1
```

Fail closed if either new r4 path already exists. Do not delete or repurpose an
existing path:

```sh
test ! -e "$CUSTODY"
test ! -e "$STUDY"
cd "$REPO"
test "$(sha256sum "$PROTOCOL" | cut -d' ' -f1)" = "$PROTOCOL_SHA"
test "$(sha256sum "$METHOD_SEAL" | cut -d' ' -f1)" = "$METHOD_SHA"
```

## 1. Create a new r4 secret after the r4 freeze

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
R4_COMMITMENT=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["task_secret_commitment_sha256"])' \
  "$CUSTODY/prepare-secret-public-output.json")
test "$R4_COMMITMENT" != \
  10e3e473540d07378c37f833a9a8faf6e8de8cd0a3c0083fa53d05f2340f6a61
printf '%s\n' "$R4_COMMITMENT"
```

The `stat` command must report mode `600` and size `32`. Only the commitment
may leave custody. The explicit inequality check prevents accidental reuse of
the r3 secret; if it fails, stop without sealing or generating.

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

## 3. Generate the new r4 suite exactly once

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

The r3 public suite must not be copied or referenced.

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
sealed, the runner and analyst must not be able to traverse the r4 custody
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
      --python-executable "$PYTHON"
  done
done
```

Any process failure outside the harness's totalized semantics aborts r4. Keep
all partial bytes and freeze a new revision; do not restart r4.

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

Publish the r4 secret, reveal, rejection log, custody/provider seals, public
bundle inventory, reveal-free analysis, and the complete r3 abort record.

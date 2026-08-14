# ExeDec V2 operations-evidence quarantine

This directory preserves an auxiliary operations-evidence packaging failure.
It is deliberately separate from the successfully verified and imported study
archive.

## Original frozen failure

The original five-file transfer is retained byte-for-byte under
`raw-transfer/`. Its operations-evidence archive and transfer seal have
SHA-256 values:

- archive:
  `aa5b9fa4a8b60ab318030cfb3ff98882de5889b5200a96ffb1e6ca85a8d96b0a`
- seal:
  `72cc4bb3998380eb12f78e15b7e17f04b6b676adf059b5efe6521ef804150ebe`

The frozen operations-evidence verifier correctly rejected the archive with:

```text
operations evidence archive safe mode differs: files/operations/bin/exedec_v2_driver.py
```

All 149 manifested payload hashes and all scientific/operational
cross-bindings independently passed. The defect was systematic packaging on a
shared FUSE mount: effective source modes `0555`, `0666`, and `0444` were
sealed as tar modes `0777` and `0666`, while the frozen verifier required
`0700` and `0600`. This is not evidence of a scientific-payload mismatch, but
it is a real failure of the frozen packaging contract.

## Local remediation-qualified derivative

After a separately frozen method and independent pre-execution audit, one
local derivative was created under `mode-header-normalized-derivative-v1/`.
It retains all 150 member paths and every payload byte and changes only tar
header modes to 146 files at `0600` and four executables at `0700`.

- derivative archive SHA-256:
  `cbb65b605b5212f521558249fe2036d8820b88056a4750ea280e5775e1f14393`
- remediation report SHA-256:
  `49e5f747b5dd78bde1e3a008205c139b5ad2ff8300602b3ede327db08af050d3`
- remediation seal SHA-256:
  `ebd74c77439879be78983bc98bfa7d5301cf30c7d3fb8ad4850f4d63926e3884`
- frozen remediation protocol SHA-256:
  `599795c487c2edbb495a10e3f9ec217d1fad5b0635458403112c14eca6f56b83`
- frozen remediation method-seal SHA-256:
  `76cf96302f0b15387218dcb8f7313c19f4772f6c9534adce788eb6a5001d5b0c`

The derivative and its two JSON records are explicitly noncanonical. They are
preserved in place and are not extracted or imported as the original
operations-evidence target. The original failed archive and failure remain
authoritative. Creating an extracted derivative would require a distinct,
frozen, remediation-qualified importer and target name.

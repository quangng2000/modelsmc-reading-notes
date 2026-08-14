# ExeDec V2 post-import audit

This is an unsealed, post-import audit note. It is not part of the frozen
study archive and does not alter any imported study byte.

## Import boundary

- Frozen study protocol SHA-256:
  `305cd2d89fbe0918a52b8e0ea6b3c4dfce23ad04ae669c48bcc3fcaa37c15b7d`
- Sealed study archive SHA-256:
  `5e97831610852264f686d0f37d6f9c4aef8d783e45c29e6cc3fd5023dbc7e46c`
- Export inventory SHA-256:
  `b7c400f6c5f076a383959f0dc2551aa057437cd413b177d49adbc3e0966780a8`
- Postrun receipt SHA-256:
  `abf929c4590535241f8bc77488886cf0893b67ef34d6e97889c82a250e1a5839`
- Imported files: 1,143
- Frozen verifier/importer SHA-256:
  `fdc4ab4523dcfac4da10787ce242898744911b31dee947b580f7e07ddb6bd1e8`
- Import result: `verified-and-imported`

The verifier rehashed every archive member against the export inventory and
atomically created `artifacts/exedec-deepcoder-ho-debug-benchmark-v2`. No
retry, resume, backfill, or overwrite was performed.

## Authoritative analysis

- Schema: `exedec-deepcoder-ho-paired-smc-debug-analysis-v2`
- Analysis SHA-256:
  `5bc9a74a39add1a60d8d5adedad89beb9dad389bd4e6cbdb4447b45e1888c2bb`
- Analysis inventory SHA-256:
  `848e204685c8b4459bb51c7471d5c644464f3183af142f6cecc60feb85a65b25`
- Integrity checks: passed
- Complete runs / paired blocks: 64 / 32
- Logical slots: 1,856 (29 per run)
- Provider calls: 188 LLM, 0 grammar-random (frozen cap 224)
- Stored-probe exact successes: 17/32 LLM-SMC and 3/32 grammar-random
- Paired cells: 16 LLM-only, 2 grammar-only, 1 both, 13 neither
- Exact two-sided conditional sign-test value: 0.001312255859375
- Public-example exact successes: 21/32 LLM-SMC and 5/32 grammar-random

## Claim boundary

This is a descriptive adapter debug, not confirmatory evidence. It uses four
deduplicated released and potentially contaminated targets with eight repeated
seed blocks per target. The strict Filter-then-Map adapter is not an official
ExeDec split or the full DeepCoder DSL. Stored private probes are finite debug
checks, not proof of arbitrary semantic equivalence or external
generalization. The exact two-sided sign-test value is reported descriptively
and must not be presented as population-level significance. Equal logical
slots do not imply equal provider, physical, or wall-clock compute.

The remote staging filesystem was shared FUSE with permissive effective modes,
so exclusive operating-system custody cannot be claimed. Hash, receipt, and
cross-binding validation passed; that integrity evidence does not remove the
custody limitation.


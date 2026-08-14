# R5 post-unblind audit

Status: **PASS — no discrepancy found.**

This audit was performed only after the reveal-free analysis and public artifact bundle had been sealed. It is a post-unblind record and is not part of the pre-unblinding public bundle. It intentionally omits the raw suite secret, private reveal, commitment nonces, rejection logs, and hidden target ASTs.

## Binding and reproducibility checks

- Study protocol SHA-256: `36bd14082c767f47327b73a1ee57f44763fcd97b2bea6dcca0637a8cddc83d2d`
- Method-seal SHA-256: `5238e17436c3e2860e2e4ee20869900e77b3534468d01a1206069e07bf49e228`
- Custody-seal SHA-256: `5980609510079cb85ce54f2bc34c75c265ee8a9e26afdc1a368accce29a39427`
- Task-secret commitment SHA-256: `0564bec1d2f0a17b0929ea94da379aa2285f4a6bfcbb82880049790e10077684`
- Public-manifest SHA-256: `d66a9437dec8de807594b2d51aa9515d48ccb0b81bd7416df027a57b363c2d5f`
- Private-reveal SHA-256: `db9f77757ddb890980caee36308a1ca1b024afae297196b86e49a9a981091efd`
- Provider-seal SHA-256: `22059898fedef2284f97f786680972a78e11408dc226c44b9603db0c8a2b527d`
- Reveal-free analysis SHA-256: `f73ef1ac255d783d912fcd8b2fc6555b9abc57efb04667490bbdd2ca6fe87196`
- Public-bundle SHA-256: `8cf0acec0b3ccedb3deeefa568a92d09cd03ff5ea3ff5f4b4a36394564e3d896`

The revealed 32-byte secret matched the reveal record and recomputed the same secret commitment in the custody seal, public manifest, and provider seal. The reveal digest matched the hidden-target-manifest digest sealed before provider calls. All 12 distinct target commitments recomputed from their canonical preimages.

Running the frozen verifier returned `exact=true` for all 12 tasks. Fresh deterministic generation from the revealed secret reproduced all 13 public JSON files (12 tasks plus manifest) and the private reveal byte-for-byte. All 415 entries in the pre-unblinding public-bundle checksum inventory passed.

## Oracle and semantic checks

The regenerated hidden programs reproduced all 144 public examples exactly. These include 96 singleton examples covering every integer in the frozen domain `{-3,-2,-1,0,1,2,3,4}`; the remaining examples are the empty list, complete ordered domain, one secret-derived permutation, and two concatenated secret-derived permutations for each task.

The 24 arms executed exactly 696 logical complete-program slots: 348 LLM-guided and 348 grammar-random. The logs contain 167 zero-loss slot occurrences representing 26 unique task-program syntaxes. Independent itemwise evaluation found every occurrence equivalent to its hidden target on all eight domain values and therefore on every list over that finite domain. Of the 12 successful arms, 4 first-exact programs were syntax-identical to the hidden target and 8 were semantic aliases; 7 successful arms visited the exact hidden syntax at some point.

This establishes finite-domain semantic recovery only. It does not establish unique hidden-AST recovery or equivalence over integers outside the frozen domain.

## Sealed outcome and resource statistics

- Slot-29 exact discovery: LLM `10/12`; grammar-random `2/12`.
- Paired discordant wins: LLM-only `8`; grammar-random-only `0`.
- Paired advantage: `+8`; one-sided exact sign-test `p=0.00390625`.
- Slot-21 exact discovery (descriptive): LLM `8/12`; grammar-random `2/12`.
- First-exact slots by task, LLM: `29, 19, 22, 11, 14, 14, —, —, 6, 9, 9, 8`.
- First-exact slots by task, grammar-random: `—, —, —, —, —, 18, —, —, —, —, —, 16`.
- Provider calls: LLM `58`; grammar-random `0`.
- LLM calls by task: `7, 5, 7, 3, 5, 5, 7, 7, 3, 3, 3, 3` (mean `4.833`; median `5`).
- Provider tokens: `137,759` total (`98,194` prompt; `39,565` completion).
- Physical scorer calls: `367`; score-cache hits: `329`.

## Claim limits

The result concerns only this frozen target distribution, model revision, semantic domain, and 29-slot budget. It supports bounded exact-discovery advantage over the matched grammar-random local-proposal arm; it does not show exhaustive-search dominance or an asymptotic speedup. Importance accounting applies to the declared weighted SMC target and does not itself guarantee discovery. The provider was not supplied the joint program catalog or its cardinality, although it could derive facts from the public grammar. The two failed tasks and all syntactic aliases remain part of the fixed-denominator record.

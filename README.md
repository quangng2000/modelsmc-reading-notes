# ModelSMC-PBE: accountable program synthesis

This repository now centers on one reproducible study: a probability-accountable
Sequential Monte Carlo (SMC) system for bounded programming by example. The
publisher-facing manuscript, code, frozen protocols, completed evidence, and
failure history are kept together so that the claims can be checked against the
artifacts that support them.

The current paper is
[ModelSMC-PBE: Probability-Accountable LLM-Shortlist SMC for Typed Programming by Example](papers/01-modelsmc/modelsmc-pbe-python/paper/main.pdf).
Its source and paragraph-level review map are in the adjacent
[paper directory](papers/01-modelsmc/modelsmc-pbe-python/paper/README.md).

## What the final paper claims

The method uses execution evidence to construct a finite typed shortlist,
totalizes proposal failures with an explicit four-slot law, evaluates the
proposal probability used by the application, and applies SMC weighting and
resampling on a bounded Filter-then-Map grammar.

The completed evidence has three roles:

- Fresh-blind r5 is the primary bounded-discovery study: the complete
  LLM-shortlist SMC system found 10 of 12 frozen targets, compared with 2 of 12
  for the matched grammar-random acquisition arm.
- Provider-free calibration V1 is a negative result: its frozen target-mass
  gate failed at 256 particles. Terminal diagnostic V2 verifies its
  finite-state identities and isolates poor terminal proposal-target overlap
  as a material mechanism, but it does not rescue V1 or validate the full SMC
  recurrence.
- ExeDec V2 is a released-data debug study, not confirmation: hidden-probe
  exactness was 17/32 paired seed blocks for LLM-SMC and 3/32 for
  grammar-random on an unofficial strict Filter-then-Map adapter to only four
  public, potentially contaminated targets.

The paper does not claim a calibrated LLM posterior, an isolated LLM-only
effect, arbitrary-task generalization, semantic equivalence from finite probes,
or compute superiority.

## Navigate the repository

- [Package landing page](papers/01-modelsmc/modelsmc-pbe-python/README.md):
  method, evidence ledger, install, and verification.
- [Publication experiment plan](papers/01-modelsmc/modelsmc-pbe-python/research/PUBLICATION_EXPERIMENT_PLAN.md):
  study-by-study claim policy and submission blockers.
- [Research evidence index](papers/01-modelsmc/modelsmc-pbe-python/research/README.md):
  current protocols, analyzers, artifacts, and preserved history.
- [Design document](papers/01-modelsmc/modelsmc-pbe-python/DESIGN.md):
  implementation and probability-accounting boundary.
- [Root artifact tree](artifacts): imported study outputs, analyses, seals, and
  retained failures. Frozen directories are evidence, not scratch space.
- [Root research bindings](research): frozen r5 protocol and runbook copies
  used at the repository-level custody boundary.

Earlier reading-note framing and superseded experimental branches remain in
version history and in their frozen files where integrity requires it. They are
not parallel publication narratives.

## Local verification

Python 3.12 and `uv` are expected. From the repository root:

```bash
cd papers/01-modelsmc/modelsmc-pbe-python
uv sync --dev
uv run pytest -q
uv run ruff check src tests research
```

Build the manuscript with Tectonic or the JMLR-compatible `latexmk` command
documented in the [paper README](papers/01-modelsmc/modelsmc-pbe-python/paper/README.md).
Running the test suite does not contact a model provider. Completed frozen
studies should not be resumed or rerun merely to reproduce the reported
analysis.

## Collaboration

Use focused changes, preserve frozen protocols and artifacts byte-for-byte, and
separate new exploratory work from completed evidence. See
[CONTRIBUTING.md](CONTRIBUTING.md) for review conventions.

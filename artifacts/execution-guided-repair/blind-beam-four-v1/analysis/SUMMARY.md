# Blind Beam Four-Task Exploratory Pilot

This report uses only the public task manifest and sealed run artifacts. It does not read the private target reveal.

## Outcome

- Primary checkpoint, slot 29: LLM exact 2/4; mean matched-random rate 2.500e-05; mean paired advantage +0.5000; plug-in P(random successes >= observed) = 0.0000.
- Secondary checkpoint, slot 37: LLM exact 2/4; mean matched-random rate 2.500e-05; mean paired advantage +0.5000; plug-in P(random successes >= observed) = 0.0000.

## Task results

- blind-01: slot 29: LLM exact (best loss 0), random 0/10000 [0.0000, 0.0004]; slot 37: LLM exact (best loss 0), random 0/10000 [0.0000, 0.0004].
- blind-02: slot 29: LLM not exact (best loss 17), random 0/10000 [0.0000, 0.0004]; slot 37: LLM not exact (best loss 17), random 0/10000 [0.0000, 0.0004].
- blind-03: slot 29: LLM not exact (best loss 25), random 1/10000 [1.765e-05, 0.0006]; slot 37: LLM not exact (best loss 25), random 1/10000 [1.765e-05, 0.0006].
- blind-04: slot 29: LLM exact (best loss 0), random 0/10000 [0.0000, 0.0004]; slot 37: LLM exact (best loss 0), random 0/10000 [0.0000, 0.0004].

## Interpretation

The paired advantages and Poisson-binomial tails are descriptive mechanism checks. Four tasks cannot establish a general search speedup, and proposal slots are not wall-clock measurements.

## Public bindings

- Manifest SHA-256: `e312e1f0c540e766aa58c2739b8ee10304290676c138a41b22d3db4b225a78e4`
- Harness SHA-256: `96954f74754ecc7713736bf2b3dac3f101740f6d21e83b60cc04abc31d3ae769`
- Study protocol SHA-256: `c92f44017ceb78f35cc3eabfce23c6babff94646ab63cc0babc92125f26e2acc`

# Research provenance

This experimental package is informed by:

- [A Probabilistic Framework for LLM-Based Model
  Discovery](https://arxiv.org/abs/2602.18266); and
- the authors' [ModelSMC reference implementation](https://github.com/mackelab/ModelSMC).

The upstream repository is used as a behavioral and architectural reference.
It is not installed as a dependency, copied into this package, or vendored
here. Upstream also identifies licensing exceptions for some of its components;
consult that repository before reusing its code or artifacts.

This package is an independent, standalone PBE adaptation. Its `paper-search`
mode must not be presented as a calibrated posterior sampler: the black-box
proposal density is unknown and no importance correction is applied. Its
`grammar-smc` mode has a known target only within the complete declared finite
skeleton.

The pure-Python program semantics, inference shell, soft loss, provider
integration, and observability code are tested but not formally verified.

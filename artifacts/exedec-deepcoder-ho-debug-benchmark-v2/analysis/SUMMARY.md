# ExeDec DeepCoder-HO Paired SMC Debug Benchmark V1

This is a debug-only result on public released targets. It is not blind, 
confirmatory, contamination-free, or an official ExeDec benchmark result.

## Hidden semantic endpoint

- LLM-SMC: 17/32
- Grammar-random: 3/32
- Paired difference: 0.4375
- Discordant blocks: 18
- Exact two-sided sign-test p: 0.00131226

Every run used 29 logical complete-program slots. Private debug oracles were 
opened only by the analyzer after all 64 public-side run artifacts validated.

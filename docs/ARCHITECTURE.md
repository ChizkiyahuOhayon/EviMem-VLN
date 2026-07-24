# Architecture and integration boundary

## Baseline-preserving design

The only authorized memory interface is:

```text
observations
  → world-key packets
  → fixed resident ledger
  → agent-centric 80×80 memory grid
  → deterministic token selector
  → original GA-VLN pos_encoding / mm_projector / memory token path
```

The policy, prompt, tokenizer, action parser, front-view tokens, and evaluator
remain baseline-owned.

## Current package

`evimem.phase0` is an asset-free reference implementation. It deliberately
avoids importing Torch, Transformers, or Habitat at package import time, which
makes core contracts testable in CI.

The reference core covers:

1. observation lifecycle and bounded storage;
2. world-cell packet updates and deterministic admission;
3. rasterization, evidence incidence, and output order;
4. corruption traces and evaluator-only metadata separation;
5. manifests and resume safety.

## Heavy-model hook policy

Actual GA-VLN hooks are enabled only after:

1. `evimem doctor` passes;
2. official baseline G0 reproduces;
3. A0 fixtures are captured;
4. Torch BF16 parity tolerances are exercised in the staged environment.

This sequencing prevents an unverified memory rewrite from being mistaken for a
scientific effect. Gaussian, learned value scoring, LoRA, and RxR policy changes
are outside Phase 0.

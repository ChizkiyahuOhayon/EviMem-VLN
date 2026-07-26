# Architecture and production integration

`GAVLNForCausalLM.encode_images` is the single memory integration boundary:

```text
SigLIP/VGGT patches + world coordinates + current pose
  → MemoryBackend.build_tokens
  → occupied agent-centric BEV features and flat cell indices
  → original GA-VLN positional encoding
  → original mm_projector
  → original <memory> token / Qwen path
```

`gavln` reconstructs the upstream windowed, agent-centric 80×80 scatter mean.
`evimem` first updates a persistent world-coordinate ledger, then rasterizes its
resident cells into the same agent-centric 80×80 readout. The downstream model
is shared; there are not two copies of GA-VLN.

## EviMem lifecycle

At every GA-VLN observation/generation event, `gavln/eval_runtime.py` supplies:

- selected SigLIP and VGGT features;
- corresponding patch world coordinates;
- monotonic observation IDs and executed-action steps;
- current agent position and rotation;
- evaluator environment ID.

Each environment owns one backend instance. `reset_episode` clears its ledger
and language cache. `reset_for_env`, used for periodic dialogue refresh, clears
only the language cache. Consequently, two distributed environments and two
successive episodes cannot share memory, while evidence survives dialogue
boundaries inside one episode.

## Ledger and readout

World points are quantized with floor division at the configured cell size.
All patches that hit the same cell in one observation are averaged before one
support update. A BLAKE2b-63 hash over seed, episode ID, and world cell gives
order-independent bottom-k admission into fixed resident storage. Existing
cells accumulate a running feature mean and support count.

Finite horizons expire a cell when `current_action - last_action > horizon`;
`route` retains it until episode reset. At readout, active world cells are
transformed by the current pose, scatter-averaged into the GA-VLN BEV grid, and
selected deterministically under `memory_token_budget`.

The reference NumPy contracts under `evimem.phase0` remain asset-free for
testing. `evimem.torch_backend.EviMemTorchBackend` is the production
implementation used by the model.

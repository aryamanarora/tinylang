# Claude Code Context for tinylang

## Project Overview
Mechanistic interpretability research comparing how different sequence model architectures (attention, Mamba, Based, DeltaNet, etc.) learn algorithms on synthetic languages (PCFGs, autoregressive tasks). Related paper: https://arxiv.org/abs/2505.15105

## Key Architecture
- **Model wrapper**: `tinylang/model/zoology.py` - wraps the Zoology framework
- **Model internals**: `tinylang/model/arch/zoology.py` - `LanguageModel` > `LMBackbone` > `backbone.layers` (Sequential of blocks)
- **Each block** has: `sequence_mixer` (attention/mamba/etc) and `state_mixer` (MLP/GLU)
- **Mixer types**: attention, mamba, mamba2, based, delta_net, gated_delta_net, h3, hyena, base_conv

## Evaluators (`tinylang/eval/`)
- **LCAEvaluator** (`lca.py`): Loss Change Attribution - tracks `grad × Δweight` and `grad × Δactivation` per training step
  - Supports `interpolation_steps` for integrated gradients
  - Hooks: embedding, layer_N, layer_N.seq_mix_in/out, layer_N.state_mix_in/out
  - Plots: heatmaps (faceted by example), trajectory plots, GIF animation
  - Logs to `all_eval_stats` for wandb
- **InterchangeEvaluator** (`interchange.py`): Activation patching with pyvene
- **SummaryEvaluator**, **AttentionEvaluator**, **ProbeEvaluator**

## Development
- Use `uv run` instead of `python` directly
- Configs in `experiments/configs/`

## Known Issues
- Commit `ab18751` added unconditional `model.to(dtype=torch.bfloat16)` which can hurt non-deltanet models
- The bf16 cast was needed for FLA/DeltaNet triton kernels but should be conditional

## Model Hooking Pattern
```python
backbone = model.model.backbone
backbone.embeddings  # embedding layer
backbone.layers[i]   # block i
backbone.layers[i].sequence_mixer  # attention/mamba/etc
backbone.layers[i].state_mixer     # MLP
```

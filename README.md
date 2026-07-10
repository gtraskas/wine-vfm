# wine-vfm

A multi-agent system that estimates **value-for-money (VFM)** for wines from
their tasting notes and metadata, and routes the estimate through a small
ensemble of specialist agents behind a chat/UI front end.

## What is VFM?

VFM is a single 0-99 score capturing how much quality a wine delivers per
dollar spent. It's derived deterministically from a wine's critic score and
price using a fixed logarithmic transform, scaled onto fixed analytic bounds
(not fitted to the data) so the metric is reproducible from constants alone.

```text
vfm = compute_vfm(points, price)
```

See [`utils/vfm.py`](utils/vfm.py) for the exact formula. The frontier RAG
agent estimates critic score and price, then maps them through this frozen
transform; the fine-tuned specialist predicts VFM directly. The bargain /
fair / overpriced `verdict()` banding lands here once the UI needs it.

## Dataset

Wine tasting notes and metadata, curated to:

- Critic score 80-100
- Price $4-250
- Tasting note length 100-2000 characters
- Deduplicated

## Architecture

A ladder of models estimates VFM directly from a wine's tasting note and
metadata:

- Baseline heuristics (random / constant / feature-based linear regression)
- Bag-of-words linear regression, random forest, and gradient-boosted tree
  variants
- A residual deep neural network
- A retrieval-augmented frontier LLM agent
- A fine-tuned open-weights specialist agent (QLoRA)
- An ensemble agent that blends the above

## Project layout

```text
utils/          # Wine model, deterministic text assembly (grows as agents need more)
agents/         # base Agent, Preprocessor, SpecialistAgent (grows day by day)
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## License

MIT — see [LICENSE](LICENSE).

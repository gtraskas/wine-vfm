# wine-vfm

A multi-agent system that finds wine bargains. It estimates **value-for-money
(VFM)** from tasting notes using a mix of models — a fine-tuned LLM, a
RAG-grounded frontier LLM, and a neural network — then scans a real online
wine shop, judges each listing, and alerts you when the price beats what the
wine's profile normally delivers.

## What is VFM?

A single 0-99 score for how much quality a wine delivers per dollar. It's
computed with a fixed formula from two numbers — the critic score and the
price:

```text
vfm = compute_vfm(points, price)
```

Nothing is fitted or learned in this formula; it's reproducible from constants
alone. See [`utils/vfm.py`](utils/vfm.py). Roughly: 70+ is a bargain, 35 or
less is overpriced.

## What makes this "agentic"?

The working definition: **LLMs equipped with tools, calling those tools in a
loop, to achieve a goal.** In practice, agentic systems share a few hallmarks,
and this project has all of them:

- **A big problem broken into small steps**, each handled by its own model
  call — scanning, estimating, deciding, notifying are separate agents here.
- **Tool calling** — an LLM is given functions it may invoke; some of those
  functions call other LLMs. That's how one model orchestrates others.
- **Structured outputs** — the model's reply is forced into a typed object
  instead of free text. Tools and structured outputs look different but are
  close cousins: most providers implement structured outputs *as* a tool
  call, and a structured reply describing "what should happen next" can
  substitute for a tool call entirely.
- **A shared environment** — a construct that lets agents pass information
  to each other. Here it's the framework holding the vectorstore, and a
  memory file both the scanner and the UI read.
- **A planning agent** — the thing that decides what happens when. It can be
  plain code (`PlanningAgent`) or an LLM given tools that wrap the other
  agents (`AutonomousPlanningAgent`, where the model doesn't even know
  its tools are agents).
- **Autonomy and memory** — the system persists beyond one chat: it
  remembers what it already surfaced, runs in the background, and reaches
  out proactively with push notifications.

## The agents

| Agent | What it does |
|---|---|
| SpecialistAgent | Fine-tuned Llama 3.2 3B that predicts VFM directly; served on Modal (GPU cloud) |
| FrontierAgent | Finds the 5 most similar wines in a vector database, shows them to gpt-5.1, gets a points+price estimate, converts to VFM |
| NeuralNetworkAgent | A small local neural network trained on bag-of-words features |
| EnsembleAgent | Blends the three estimates with weights fitted by linear regression (not hand-picked) |
| ScannerAgent | Pulls live listings from a real wine shop, ranks them by VFM, has gpt-5-mini extract the 5 best-documented critic-scored wines |
| MessagingAgent | Sends push notifications (Pushover); logs instead if no keys configured |
| PlanningAgent | Plain-code orchestrator: scan → estimate each wine → alert the best if it clears a threshold |
| AutonomousPlanningAgent | The LLM version: gpt-5.1 gets scan/estimate/notify as tools and runs the workflow itself — this one drives the web app |

## Progress — the notebooks

**[1.ipynb](1.ipynb) — Deploy the specialist.** Set up Modal, deploy the
fine-tuned model behind a stable cloud endpoint, wrap it as SpecialistAgent.
Result: `agent.estimate(wine_text)` returns a VFM score from the cloud GPU.

**[2.ipynb](2.ipynb) — RAG and the ensemble.** Load the curated dataset
(~88K wines), embed every tasting note into a Chroma vector database,
visualize it in 3D. Build the FrontierAgent (RAG), retrain the neural
network, then fit the ensemble weights by regression on 200 validation
wines. Result: three working estimators, evaluated with MAE/R², plus two
saved artifacts (`vfm_net.pth`, `ensemble_model.joblib`).

**[3.ipynb](3.ipynb) — Scan the real world.** Pull live listings from an
online wine shop, keep only wines with a printed critic score, rank them by
VFM (score and price are both printed — no model needed), and let gpt-5-mini
extract the 5 best into typed objects. Result: `agent.scan()` returns five
genuine bargains, e.g. a 92-point Sauvignon Blanc at $8.99 (VFM 78).

**[4.ipynb](4.ipynb) — The agentic loop.** First with fake tools to see the
mechanics: the LLM asks for a tool, code runs it, the result goes back, the
loop repeats until the model answers in plain text. Then the real
AutonomousPlanningAgent: gpt-5.1 scans the shop, gets ensemble estimates,
picks the single best bargain, and notifies. Result: `agent.plan()` returns
an Opportunity — the chosen wine with its actual-vs-estimated VFM.

**[5.ipynb](5.ipynb) — The web app.** A Gradio UI built piece by piece, then
the real thing in [`app.py`](app.py): one button. Press it and one autonomous
run happens live — the agents' color-coded logs stream into the page while
the LLM planner scans the shop and judges candidates. A verdict card shows
the latest find (bargain / fair / overpriced), and a table accumulates every
wine surfaced so far. Result: `uv run app.py` opens the app in your browser.

## Dataset

Wine tasting notes and metadata, curated to critic score 80-100, price
$4-250, note length 100-2000 characters, deduplicated. ~88K train / 11K
validation / 11K test, hosted on the HuggingFace Hub.

## Project layout

```text
utils/                      # Wine model, VFM formula, text assembly, evaluator, listings
agents/                     # all the agents listed above
wine_agent_framework.py     # ties it together: vectorstore, memory.json, planner
app.py                      # the Gradio web app — uv run app.py
specialist_service2.py      # the Modal deployment of the fine-tuned model
1.ipynb ... 5.ipynb         # the day-by-day build, in order
```

## Setup

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Needs a `.env` with `OPENAI_API_KEY` and Modal tokens (`MODAL_TOKEN_ID`,
`MODAL_TOKEN_SECRET`). Optional: `PUSHOVER_USER`/`PUSHOVER_TOKEN` for real
push notifications.

## License

MIT — see [LICENSE](LICENSE).

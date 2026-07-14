"""Web UI: press the button, watch the agents hunt one wine bargain live.

Unlike a continuously-scheduled hunter, this app runs on demand: one press
triggers one autonomous planning run (scan the shop, estimate each candidate
with the ensemble, alert the best find). Agent logs stream into the page
while the run progresses, and the headline card shows the latest find,
backed by the framework's persistent memory.

Run with: uv run app.py
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Iterator

from dotenv import load_dotenv
import gradio as gr
from gradio.themes import Default

from utils.listings import Opportunity
from utils.log_utils import reformat
from utils.vfm import verdict
from wine_agent_framework import WineAgentFramework

load_dotenv(override=True)

# Define Theme Colors (Light Theme)
BACKGROUND = "#ffffff"
PANEL = "#f8f9fa"
TEXT = "#1a1a24"

# 1. Native Theme Definition mapped to light colors across both configurations
custom_theme = Default().set(
    # --- Light Mode variables ---
    body_background_fill=BACKGROUND,
    body_text_color=TEXT,
    block_background_fill=PANEL,
    block_border_color="#cccccc",
    background_fill_primary=BACKGROUND,
    background_fill_secondary=PANEL,

    # --- Dark Mode variables (Forced to Light to prevent browser style leakage) ---
    body_background_fill_dark=BACKGROUND,
    body_text_color_dark=TEXT,
    block_background_fill_dark=PANEL,
    block_border_color_dark="#cccccc",
    background_fill_primary_dark=BACKGROUND,
    background_fill_secondary_dark=PANEL,

    # Sizing adjustments (larger fonts)
    block_title_text_size="20px",
    body_text_size="16px",
)

# 2. CSS Overrides for modal canvas alignment and log console isolation
CSS = f"""
footer {{visibility: hidden}}

/* Force white backgrounds on both Local and Modal */
body, html, .gradio-container {{
    background-color: {BACKGROUND} !important;
}}

/* Ensure all primary text is dark and legible */
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container p, .gradio-container strong {{
    color: {TEXT} !important;
}}

/* Global paragraph font-size normalization */
.gradio-container p, .gradio-container li, .gradio-container span {{
    font-size: 16px !important;
}}

/* Block formatting */
.gr-block, .gr-box, .gr-form, .gr-panel {{
    background-color: {PANEL} !important;
    border-color: #cccccc !important;
}}

/* ==========================================
   STRICT ISOLATION FOR LOG PANEL
   This remains 100% untouched and renders
   as a dark terminal regardless of the page theme.
   ========================================== */
.log-terminal, .log-terminal * {{
    color: #c8c8d0 !important;
    font-family: monospace !important;
    font-size: 13px !important;
    background-color: #222229 !important;
}}
"""


class QueueHandler(logging.Handler):
    """Forwards log records into a queue the UI generator drains."""

    def __init__(self, log_queue: queue.Queue[str]) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        self.log_queue.put(self.format(record))


def html_for(log_data: list[str]) -> str:
    """Render the last log lines as a scrollable dark panel."""
    output = "<br>".join(log_data[-18:])
    return (
        '<div class="log-terminal" style="height: 400px; overflow-y: auto; border: 1px solid #444; '
        "background-color: #222229; color: #c8c8d0; padding: 10px; "
        f'font-family: monospace; font-size: 13px;">{output}</div>'
    )


def attach_log_handler(log_queue: queue.Queue[str]) -> logging.Handler:
    """Attach a queue-backed handler to the root logger for one run."""
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def clean_title(title: str) -> str:
    """Strip critic shorthand like [WE92][JS90] from a shop title."""
    return re.sub(r"\s*\[[^\]]*\]", "", title).strip()


VERDICT_LABELS = {"bargain": "A BARGAIN", "fair": "FAIRLY PRICED", "overpriced": "OVERPRICED"}


def headline_for(opportunities: list[Opportunity]) -> str:
    """Markdown card describing the latest find, written for a casual reader."""
    if not opportunities:
        return "*No finds yet — press the button and watch the agents work.*"
    latest = opportunities[-1]
    listing = latest.listing
    label, _ = verdict(latest.actual_vfm)
    return (
        f"### Latest find: [{clean_title(listing.title)}]({listing.url})\n"
        f"**{listing.points} points at ${listing.price:.2f} — {VERDICT_LABELS[label]}.**\n\n"
        f"Wine critics scored this wine {listing.points} out of 100. At "
        f"${listing.price:.2f} a bottle, that quality-for-money works out to "
        f"**{latest.actual_vfm} on our 0-99 value scale**.\n\n"
        f"Our models read only the wine's profile — the tasting notes, grape variety, "
        f"country, region and winery, but never the price — and expected a wine like this "
        f"to sit around {latest.estimated_vfm}. {delta_sentence(latest.delta)}"
    )


def delta_sentence(delta: int) -> str:
    """One casual sentence interpreting the actual-vs-expected gap."""
    if delta > 0:
        return (
            f"The shop's price beats that expectation by **{delta} points** — a better "
            "deal than the description alone would suggest."
        )
    if delta == 0:
        return "The shop's price lands exactly on that expectation."
    return f"The shop's price falls **{-delta} points** short of that expectation."


class App:
    """Owns the lazily-created framework and builds the Gradio UI."""

    def __init__(self) -> None:
        self.framework: WineAgentFramework | None = None

    def get_framework(self) -> WineAgentFramework:
        if not self.framework:
            self.framework = WineAgentFramework()
        return self.framework

    def hunt(self, log_data: list[str]) -> Iterator[tuple[list[str], str, str]]:
        """One button press: run the agents in a thread, stream logs live."""
        log_queue: queue.Queue[str] = queue.Queue()
        result_queue: queue.Queue[list[Opportunity]] = queue.Queue()
        handler = attach_log_handler(log_queue)

        def worker() -> None:
            result_queue.put(self.get_framework().run())

        thread = threading.Thread(target=worker)
        thread.start()

        memory = self.get_framework().memory
        final: list[Opportunity] | None = None
        try:
            while final is None or not log_queue.empty():
                try:
                    message = log_queue.get_nowait()
                    log_data.append(reformat(message))
                    current = final or memory
                    yield log_data, html_for(log_data), headline_for(current)
                except queue.Empty:
                    try:
                        final = result_queue.get_nowait()
                    except queue.Empty:
                        time.sleep(0.1)
            yield log_data, html_for(log_data), headline_for(final)
        finally:
            logging.getLogger().removeHandler(handler)

    def build_ui(self) -> gr.Blocks:
        """Build the UI — launched locally by run(), or mounted by the deployment."""
        initial_memory = self.get_framework().memory
        with gr.Blocks(title="Wine Value Hunter", theme=custom_theme, css=CSS) as ui:
            log_data = gr.State([])

            gr.Markdown("# Wine Value Hunter")
            gr.Markdown(
                "Agentic AI that finds underpriced wines in a live shop. A fine-tuned LLM "
                "on Modal, a RAG pipeline over 88K tasting notes, and a neural network "
                "collaborate under an LLM planner."
            )
            gr.Markdown(
                "Press the button: the agents scan a real "
                "wine shop, judge the value of each candidate, and surface the single best "
                "bargain for you."
            )
            with gr.Row():
                hunt_button = gr.Button("🍷 Hunt for a bargain", variant="primary", scale=1)

            gr.Markdown("A hunt takes a minute or two. Only one hunt runs at a time.")

            headline = gr.Markdown(headline_for(initial_memory))
            gr.Markdown("**Agent activity** — live logs, one color per agent")
            logs = gr.HTML(html_for([]))

            hunt_button.click(  # type: ignore[attr-defined, unused-ignore]  # dynamic event
                self.hunt,
                inputs=[log_data],
                outputs=[log_data, logs, headline],
            )

        # One hunt at a time — a second visitor queues instead of double-spending
        ui.queue(default_concurrency_limit=1)
        blocks: gr.Blocks = ui
        return blocks

    def run(self) -> None:
        """Launch the UI on a local server."""
        self.build_ui().launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()

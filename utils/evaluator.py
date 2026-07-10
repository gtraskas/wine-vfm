"""Evaluation harness: MAE ± 95% CI, R², color-coded scatter, error trend.

Tuned for the 0-99 VFM scale: absolute error bands, and R² reported
alongside MAE. Every model runs through this same harness on the same
held-out slice, so results are directly comparable.
"""

import math
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate

import numpy as np
import plotly.graph_objects as go

from utils.items import Wine

GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

DEFAULT_SIZE: int = 200
DEFAULT_WORKERS: int = 10

# Absolute bands for the 0-99 VFM scale
GREEN_ERROR: float = 5.0
ORANGE_ERROR: float = 15.0

Predictor = Callable[[Wine], float | str]


class Tester:
    """Runs a predictor against labeled test data and reports metrics.

    Attributes:
        predictor: Callable Wine -> float | str (strings are parsed).
        data: Labeled test set.
        title: Display title, derived from the predictor name by default.
        size: Number of test items to evaluate.
        workers: Thread concurrency (set 1 if hitting API rate limits).
    """

    def __init__(
        self,
        predictor: Predictor,
        data: list[Wine],
        title: str | None = None,
        size: int = DEFAULT_SIZE,
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = min(size, len(data))
        self.workers = workers
        self.guesses: list[float] = []
        self.truths: list[float] = []
        self.errors: list[float] = []
        self.colors: list[str] = []
        self.titles: list[str] = []

    @staticmethod
    def post_process(value: float | str) -> float:
        """Extract a float from a possibly-noisy string prediction."""
        if isinstance(value, str):
            cleaned = value.replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", cleaned)
            return float(match.group()) if match else 0.0
        return float(value)

    @staticmethod
    def color_for(error: float) -> str:
        """Green/orange/red banding for the 0-99 VFM scale."""
        if error < GREEN_ERROR:
            return "green"
        if error < ORANGE_ERROR:
            return "orange"
        return "red"

    def run_datapoint(self, index: int) -> None:
        """Predict one item, record error and color."""
        wine = self.data[index]
        guess = self.post_process(self.predictor(wine))
        truth = float(wine.vfm or 0)
        error = abs(guess - truth)
        color = self.color_for(error)
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.colors.append(color)
        self.titles.append(wine.title[:40])
        print(
            f"{COLOR_MAP[color]}{index + 1}: Guess {guess:.0f} "
            f"Truth {truth:.0f} Error {error:.0f}{RESET}"
        )

    def run(self) -> None:
        """Evaluate, then render the scatter and the error-trend chart."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self.run_datapoint, range(self.size)))
        self.report()

    def report(self) -> None:
        """Print MAE with 95% CI and R², then show both charts."""
        mae = float(np.mean(self.errors))
        ci = 1.96 * float(np.std(self.errors)) / math.sqrt(len(self.errors))
        truths, guesses = np.array(self.truths), np.array(self.guesses)
        ss_res = float(np.sum((truths - guesses) ** 2))
        ss_tot = float(np.sum((truths - truths.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        hits = sum(1 for c in self.colors if c == "green")
        print(
            f"\n{self.title}  MAE: {mae:.2f} ± {ci:.2f} VFM  "
            f"R²: {r_squared:.3f}  Hits: {hits / len(self.errors):.0%}"
        )
        self._scatter(mae, ci, r_squared)
        self._error_trend()

    def _scatter(self, mae: float, ci: float, r_squared: float) -> None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 99],
                y=[0, 99],
                mode="lines",
                line=dict(color="#8A9BB0", dash="dash"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.truths,
                y=self.guesses,
                mode="markers",
                marker=dict(color=self.colors, size=6, opacity=0.7),
                text=self.titles,
                showlegend=False,
            )
        )
        fig.update_layout(
            title=f"{self.title}  MAE: {mae:.2f} ± {ci:.2f}  R²: {r_squared:.3f}",
            xaxis_title="Actual VFM",
            yaxis_title="Predicted VFM",
            width=800,
            height=500,
            template="plotly_white",
        )
        fig.show()

    def _error_trend(self) -> None:
        """Cumulative mean error with a shaded 95% CI band, as sample size grows."""
        n = len(self.errors)
        x = list(range(1, n + 1))

        running_sums = list(accumulate(self.errors))
        running_means = [s / i for s, i in zip(running_sums, x, strict=True)]

        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt(max(sq_sum / i - mean**2, 0.0)) if i > 1 else 0.0
            for i, sq_sum, mean in zip(x, running_squares, running_means, strict=True)
        ]

        ci = [
            1.96 * sd / math.sqrt(i) if i > 1 else 0.0
            for i, sd in zip(x, running_stds, strict=True)
        ]
        upper = [m + c for m, c in zip(running_means, ci, strict=True)]
        lower = [m - c for m, c in zip(running_means, ci, strict=True)]

        fig = go.Figure()

        # Shaded CI band — added first so it renders behind the mean line
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="#C47E7E"),
                customdata=ci,
                hovertemplate=(
                    "n=%{x}<br>Avg Error=%{y:.2f} VFM<br>±95% CI=%{customdata:.2f}<extra></extra>"
                ),
            )
        )

        final_mean, final_ci = running_means[-1], ci[-1]
        fig.update_layout(
            title=f"{self.title} — cumulative average error: {final_mean:.2f} ± {final_ci:.2f} VFM",
            xaxis_title="Test items",
            yaxis_title="Cumulative MAE (VFM)",
            width=800,
            height=300,
            template="plotly_white",
            showlegend=False,
        )
        fig.show()


def evaluate(
    predictor: Predictor,
    data: list[Wine],
    size: int = DEFAULT_SIZE,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Convenience wrapper: build a Tester and run it."""
    Tester(predictor, data, size=size, workers=workers).run()

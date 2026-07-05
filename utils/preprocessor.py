"""Builds the model input text for each Wine.

Deterministic template assembly — no LLM call. The fine-tuned specialist
model was trained on text built by this exact assembly, so agent-time
inference must use the same code path to avoid train/inference skew.
"""

from __future__ import annotations

from utils.items import Wine

INCLUDE_TITLE: bool = False  # title embeds vintage+winery -> memorization risk


class TextAssembler:
    """Deterministic input assembly — the only preprocessing path."""

    @staticmethod
    def assemble(wine: Wine) -> str:
        """Build the standardized input text from note + structured context.

        Args:
            wine: Wine with `full` and structured metadata populated.

        Returns:
            Flat text combining the tasting note with structured context.
        """
        lines = [wine.full or "", ""]
        if INCLUDE_TITLE:
            lines.append(f"Title: {wine.title}")
        lines.extend(
            [
                f"Variety: {wine.variety}",
                f"Country: {wine.country}",
                f"Province: {wine.province}",
                f"Region: {wine.region}",
                f"Winery: {wine.winery}",
            ]
        )
        return "\n".join(lines)

"""Agent-facing preprocessing: builds specialist model input from a Wine.

Deterministic only — no LLM rewriting. The specialist model was fine-tuned
on text assembled by utils.preprocessor.TextAssembler, so this wrapper must
call that same code path to guarantee train/inference parity.
"""

from __future__ import annotations

from utils.items import Wine
from utils.preprocessor import TextAssembler


class Preprocessor:
    """Builds standardized specialist-agent input text from a Wine."""

    def preprocess(self, wine: Wine) -> str:
        """Assemble the model input text for a wine.

        Args:
            wine: Wine with `full` and structured metadata populated.

        Returns:
            Flat text ready to pass to the specialist agent.
        """
        return TextAssembler.assemble(wine)

"""Wraps the deployed Modal specialist service (fine-tuned Llama 3.2 3B + LoRA)."""

from __future__ import annotations

import modal

from agents.agent import Agent


class SpecialistAgent(Agent):
    """Calls the deployed wine VFM specialist model on Modal."""

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self) -> None:
        """Connect to the deployed WineSpecialist class on Modal."""
        self.log("Initializing — connecting to Modal")
        specialist_cls = modal.Cls.from_name("wine-vfm-specialist-service", "WineSpecialist")
        self.specialist = specialist_cls()
        self.log("Ready")

    def estimate(self, description: str) -> int:
        """Estimate a wine's VFM score (0-99).

        Args:
            description: Text produced by utils.preprocessor.TextAssembler.

        Returns:
            Estimated VFM score.
        """
        self.log("Calling remote fine-tuned model")
        result: int = self.specialist.estimate_vfm.remote(description)
        self.log(f"Result: {result}")
        return result

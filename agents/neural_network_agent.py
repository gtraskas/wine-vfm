"""Wraps the locally-trained residual bag-of-words VFM regressor."""

from __future__ import annotations

from agents.agent import Agent
from utils.deep_neural_network import VfmTrainer

WEIGHTS_PATH = "vfm_net.pth"


class NeuralNetworkAgent(Agent):
    """Serves VFM predictions from the trained VfmNet weights."""

    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self, weights_path: str = WEIGHTS_PATH) -> None:
        """Load the trained network (see day2 notebook for the training cell).

        Args:
            weights_path: Checkpoint produced by VfmTrainer.save().
        """
        self.log("Initializing — loading network weights")
        self.trainer = VfmTrainer()
        self.trainer.load(weights_path)
        self.log("Ready")

    def estimate(self, description: str) -> int:
        """Estimate a wine's VFM score (0-99).

        Args:
            description: Assembled summary text (utils.preprocessor style).

        Returns:
            VFM score 0-99.
        """
        self.log("Running network inference")
        result = int(round(min(max(self.trainer.predict(description), 0.0), 99.0)))
        self.log(f"Result: {result}")
        return result

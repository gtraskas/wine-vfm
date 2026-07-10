"""Combines the three VFM estimators with regression-derived weights.

The weights are NOT hardcoded: a linear regression is fitted on held-out
validation wines (see the day2 notebook), with the three sub-agent
predictions as features and true VFM as the target, then persisted. This
agent loads that artifact and applies it. Feature order must match the
fitting code: [specialist, frontier, neural_network].
"""

from __future__ import annotations

import chromadb
import joblib
import numpy as np

from agents.agent import Agent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.specialist_agent import SpecialistAgent

ENSEMBLE_PATH = "ensemble_model.joblib"


class EnsembleAgent(Agent):
    """Weighted blend of specialist, frontier, and neural-network estimates."""

    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(
        self, collection: chromadb.Collection, ensemble_path: str = ENSEMBLE_PATH
    ) -> None:
        """Create the three sub-agents and load the fitted blend weights.

        Args:
            collection: Chroma collection for the frontier agent's RAG.
            ensemble_path: Fitted LinearRegression artifact (day2 notebook).
        """
        self.log("Initializing — creating sub-agents")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.model = joblib.load(ensemble_path)
        self.log("Ready")

    def estimate(self, description: str) -> int:
        """Estimate VFM as the regression-weighted blend of the sub-agents.

        Args:
            description: Assembled summary text (utils.preprocessor style).

        Returns:
            VFM score 0-99.
        """
        self.log("Running ensemble")
        features = np.array(
            [
                [
                    self.specialist.estimate(description),
                    self.frontier.estimate(description),
                    self.neural_network.estimate(description),
                ]
            ],
            dtype=float,
        )
        blended = float(self.model.predict(features)[0])
        result = int(round(min(max(blended, 0.0), 99.0)))
        self.log(f"Ensemble result: {result}")
        return result

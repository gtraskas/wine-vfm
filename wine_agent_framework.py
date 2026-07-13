"""Ties the agent system together: vectorstore, persistent memory, planner.

Memory is a JSON file of every Opportunity ever surfaced — it feeds the UI
table and, as surfaced URLs, the scanner's dedupe so reruns only find new
wines. Run directly (python wine_agent_framework.py) for one planning pass.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import chromadb
from dotenv import load_dotenv

from agents.autonomous_planning_agent import AutonomousPlanningAgent
from utils.listings import Opportunity

load_dotenv(override=True)

BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"


def init_logging() -> None:
    """Route INFO logging to stdout with a timestamped format."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class WineAgentFramework:
    """Owns the vectorstore, the opportunity memory, and the planner."""

    DB = "wine_vectorstore"
    COLLECTION = "wines"
    MEMORY_FILENAME = "memory.json"

    def __init__(self) -> None:
        init_logging()
        client = chromadb.PersistentClient(path=self.DB)
        self.collection = client.get_or_create_collection(self.COLLECTION)
        self.memory = self.read_memory()
        self.planner: AutonomousPlanningAgent | None = None

    def init_agents_as_needed(self) -> None:
        """Create the planner lazily — its sub-agents load models on init."""
        if not self.planner:
            self.log("Initializing agent framework")
            self.planner = AutonomousPlanningAgent(self.collection)
            self.log("Agent framework is ready")

    def read_memory(self) -> list[Opportunity]:
        """Load previously surfaced opportunities from disk."""
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME) as file:
                data = json.load(file)
            return [Opportunity(**item) for item in data]
        return []

    def write_memory(self) -> None:
        """Persist all surfaced opportunities to disk."""
        data = [opportunity.model_dump() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    def log(self, message: str) -> None:
        logging.info(f"{BG_BLUE}{WHITE}[Agent Framework] {message}{RESET}")

    def run(self) -> list[Opportunity]:
        """One planning pass: scan, judge, alert; append any find to memory."""
        self.init_agents_as_needed()
        assert self.planner is not None
        surfaced_urls = [opportunity.listing.url for opportunity in self.memory]
        self.log("Kicking off the planning agent")
        result = self.planner.plan(memory=surfaced_urls)
        self.log(f"Planning agent returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory


if __name__ == "__main__":
    WineAgentFramework().run()

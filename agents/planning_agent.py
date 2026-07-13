"""Deterministic orchestrator: scan -> estimate -> alert the best bargain.

For each scanned wine, the ensemble estimates the VFM its tasting profile
typically delivers; the actual VFM at the shop price comes from the printed
critic score through the frozen transform. A positive delta means the shop
price delivers more value than the profile implies — the best one is alerted
if it clears the threshold.
"""

from __future__ import annotations

import chromadb

from agents.agent import Agent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.scanner_agent import ScannerAgent
from utils.listings import Listing, Opportunity
from utils.preprocessor import TextAssembler

# Minimum actual-minus-estimated VFM delta worth alerting — tunable
DELTA_THRESHOLD = 10


class PlanningAgent(Agent):
    """Coordinates scanner, ensemble, and messenger into one bargain run."""

    name = "Planning Agent"
    color = Agent.GREEN

    def __init__(self, collection: chromadb.Collection) -> None:
        """Create the three agents this planner coordinates.

        Args:
            collection: Chroma collection for the ensemble's frontier agent.
        """
        self.log("Initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Ready")

    def run(self, listing: Listing) -> Opportunity:
        """Judge one scanned wine: actual VFM at the shop price vs estimate.

        Args:
            listing: A critic-scored wine from the scanner.

        Returns:
            The listing with its estimated/actual VFM and delta.
        """
        self.log(f"Evaluating: {listing.title[:50]}")
        summary = TextAssembler.assemble(listing.to_wine())
        estimated = self.ensemble.estimate(summary)
        actual = listing.actual_vfm()
        delta = actual - estimated
        self.log(f"Actual VFM {actual} vs estimated {estimated} (delta {delta:+d})")
        return Opportunity(
            listing=listing, estimated_vfm=estimated, actual_vfm=actual, delta=delta
        )

    def plan(self, memory: list[str] | None = None) -> Opportunity | None:
        """Run the full workflow: scan, judge each wine, alert the best.

        Args:
            memory: URLs of listings surfaced in previous runs.

        Returns:
            The best Opportunity if it clears DELTA_THRESHOLD, else None.
        """
        self.log("Kicking off a run")
        selection = self.scanner.scan(memory=memory)
        if not selection or not selection.listings:
            self.log("No new wines to evaluate")
            return None
        opportunities = [self.run(listing) for listing in selection.listings[:5]]
        opportunities.sort(key=lambda opp: opp.delta, reverse=True)
        best = opportunities[0]
        self.log(f"Best opportunity has delta {best.delta:+d}")
        if best.delta > DELTA_THRESHOLD:
            self.messenger.alert(best)
            self.log("Run complete — alerted")
            return best
        self.log("Run complete — nothing cleared the threshold")
        return None

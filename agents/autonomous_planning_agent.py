"""LLM-driven orchestrator: the model decides which tools to call and when.

Same three capabilities as the deterministic planner, exposed as tools to a
frontier model in an agentic loop. Wines are referenced between tools by
URL: the code resolves each URL back to the scanned listing and assembles
the ensemble input deterministically, so no LLM paraphrase ever reaches the
models downstream.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import chromadb
from openai import OpenAI

from agents.agent import Agent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.scanner_agent import ScannerAgent
from utils.listings import Listing, Opportunity
from utils.preprocessor import TextAssembler

MODEL = "gpt-5.1"

SYSTEM_MESSAGE = (
    "You find great value wines using your tools, and notify the user of the single "
    "best bargain."
)
USER_MESSAGE = """First, use your tool to scan the wine shop for promising listings — each \
comes with its actual VFM (value for money, 0-99) at the shop price. Then for each listing, \
use your tool to estimate the VFM its tasting profile typically delivers. Then pick the \
single wine whose actual VFM most exceeds its typical estimate, and use your tool to notify \
the user, passing that wine's url and your estimated typical VFM. Then just reply OK to \
indicate success."""


class AutonomousPlanningAgent(Agent):
    """Lets a frontier model orchestrate scanner, ensemble, and messenger."""

    name = "Autonomous Planning Agent"
    color = Agent.GREEN

    scan_function = {
        "name": "scan_the_wine_shop_for_bargains",
        "description": (
            "Returns the best-value wine listings from the shop, each with its printed "
            "critic score, shop price, URL, and the actual VFM (0-99) at that price"
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_typical_vfm",
        "description": (
            "Given a listing's URL, estimate the VFM (0-99) that this wine's tasting "
            "profile typically delivers, judged from its description alone"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the listing, exactly as returned by the scan",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_bargain",
        "description": (
            "Send the user a push notification about the single best bargain; "
            "only call this one time"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the chosen listing, exactly as scanned",
                },
                "estimated_vfm": {
                    "type": "integer",
                    "description": "The typical VFM you obtained from the estimate tool",
                },
            },
            "required": ["url", "estimated_vfm"],
            "additionalProperties": False,
        },
    }

    def __init__(self, collection: chromadb.Collection) -> None:
        """Create the three agents this planner coordinates.

        Args:
            collection: Chroma collection for the ensemble's frontier agent.
        """
        self.log("Initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.client = OpenAI()
        self.memory: list[str] = []
        self.scanned: dict[str, Listing] = {}
        self.opportunity: Opportunity | None = None
        self.log("Ready")

    def scan_the_wine_shop_for_bargains(self) -> str:
        """Tool: scan the shop and return listings with their actual VFM."""
        self.log("Tool call: scanning the wine shop")
        selection = self.scanner.scan(memory=self.memory)
        if not selection or not selection.listings:
            return "No new wines found"
        self.scanned = {listing.url: listing for listing in selection.listings}
        enriched = [
            {**listing.model_dump(), "actual_vfm": listing.actual_vfm()}
            for listing in selection.listings
        ]
        return json.dumps({"listings": enriched})

    def estimate_typical_vfm(self, url: str) -> str:
        """Tool: ensemble estimate for the listing behind this URL."""
        self.log("Tool call: estimating typical VFM via the ensemble")
        listing = self.scanned.get(url)
        if listing is None:
            return f"Unknown url {url} — use a url returned by the scan tool"
        summary = TextAssembler.assemble(listing.to_wine())
        estimate = self.ensemble.estimate(summary)
        return f"The tasting profile of {listing.title} typically delivers VFM {estimate}"

    def notify_user_of_bargain(self, url: str, estimated_vfm: int) -> str:
        """Tool: alert the user about the chosen listing, once."""
        if self.opportunity:
            self.log("Tool call: notify requested a 2nd time; ignoring")
            return "Notification already sent"
        listing = self.scanned.get(url)
        if listing is None:
            return f"Unknown url {url} — use a url returned by the scan tool"
        self.log("Tool call: notifying the user")
        actual = listing.actual_vfm()
        self.opportunity = Opportunity(
            listing=listing,
            estimated_vfm=estimated_vfm,
            actual_vfm=actual,
            delta=actual - estimated_vfm,
        )
        self.messenger.alert(self.opportunity)
        return "Notification sent ok"

    def get_tools(self) -> list[Any]:
        """Tool schemas for the chat completion call."""
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message: Any) -> list[dict[str, str]]:
        """Execute the tools requested by the model and format the results."""
        mapping: dict[str, Callable[..., str]] = {
            "scan_the_wine_shop_for_bargains": self.scan_the_wine_shop_for_bargains,
            "estimate_typical_vfm": self.estimate_typical_vfm,
            "notify_user_of_bargain": self.notify_user_of_bargain,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    def plan(self, memory: list[str] | None = None) -> Opportunity | None:
        """Run the agentic loop until the model stops requesting tools.

        Args:
            memory: URLs of listings surfaced in previous runs.

        Returns:
            The Opportunity the model chose to alert, or None.
        """
        self.log("Kicking off an autonomous run")
        self.memory = memory or []
        self.scanned = {}
        self.opportunity = None
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ]
        done = False
        while not done:
            response = self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        self.log(f"Completed with: {response.choices[0].message.content}")
        return self.opportunity

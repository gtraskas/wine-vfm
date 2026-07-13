"""Sends push notifications about wine bargains via Pushover.

Needs PUSHOVER_USER and PUSHOVER_TOKEN in the environment — free keys
from pushover.net (user key on the dashboard, token from a created
application).
"""

from __future__ import annotations

import os

import requests

from agents.agent import Agent
from utils.listings import Opportunity

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
TIMEOUT_SECONDS = 10


class MessagingAgent(Agent):
    """Pushes bargain alerts to the user's phone."""

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self) -> None:
        """Read the Pushover credentials from the environment."""
        self.log("Initializing")
        self.pushover_user = os.environ["PUSHOVER_USER"]
        self.pushover_token = os.environ["PUSHOVER_TOKEN"]
        self.log("Ready")

    def push(self, text: str) -> None:
        """Send a push notification.

        Args:
            text: Message body.
        """
        self.log(f"Pushing: {text[:60]}")
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",
        }
        requests.post(PUSHOVER_URL, data=payload, timeout=TIMEOUT_SECONDS)

    def alert(self, opportunity: Opportunity) -> None:
        """Notify about one bargain: what it is, the numbers, and the link.

        Args:
            opportunity: Scanned wine with its actual-vs-estimated VFM.
        """
        listing = opportunity.listing
        text = (
            f"Wine bargain! {listing.title} — {listing.points} pts at ${listing.price:.2f}. "
            f"VFM {opportunity.actual_vfm} vs typical {opportunity.estimated_vfm} "
            f"({opportunity.delta:+d}). {listing.url}"
        )
        self.push(text)

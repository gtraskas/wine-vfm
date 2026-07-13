"""Sends push notifications about wine bargains via Pushover.

Optional: with PUSHOVER_USER and PUSHOVER_TOKEN in the environment (free
keys from pushover.net), alerts go to your phone; without them, alerts
are logged instead — nothing downstream requires the real push.
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
        """Read the Pushover credentials from the environment, if present."""
        self.log("Initializing")
        self.pushover_user = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        if self.pushover_user and self.pushover_token:
            self.log("Ready — Pushover configured")
        else:
            self.log("Ready — no Pushover keys, alerts will be logged only")

    def push(self, text: str) -> None:
        """Send a push notification, or log it if Pushover isn't configured.

        Args:
            text: Message body.
        """
        if not (self.pushover_user and self.pushover_token):
            self.log(f"Alert (log only): {text}")
            return
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

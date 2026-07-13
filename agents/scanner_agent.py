"""Scans the wine shop for critic-scored, high-value listings.

Fetches live listings, keeps only those with a printed critic score, and
ranks them by VFM computed from the printed score and shop price — pure
Python, no model call — so only the best-value candidates reach the LLM.
The LLM then selects the five best-documented of those and extracts the
score, price, and the structured fields the rest of the system expects.
Listings without an explicit printed score are never included.
"""

from __future__ import annotations

from openai import OpenAI

from agents.agent import Agent
from utils.listings import ListingSelection, ScrapedListing

MODEL = "gpt-5-mini"
# Cap the prompt: 50 described listings is roughly 20K tokens
MAX_CANDIDATES = 50

SYSTEM_PROMPT = """You identify the 5 best-documented wines from a list of shop listings and \
extract their details. The listings are ordered by value for money, best first — prefer \
earlier listings whenever their documentation is adequate. Select only listings that have \
BOTH an explicit printed critic score (like 'James Suckling, 95 Points') AND a quoted \
tasting note. Only grape wines — never whiskey, bourbon, beer, sake, or other spirits.

For each selected wine extract: the title, the quoted tasting note (the critic's prose, \
without surrounding quotes), the critic score as an integer (if several critics are shown, \
use the first), the price as a number, the URL copied verbatim, and the variety, country, \
province, region, and winery from the description. If a field is not stated, infer it from \
the title or region hierarchy; use an empty string only as a last resort.

Respond strictly in JSON. Never include a wine whose critic score or price is unclear."""

USER_PROMPT_PREFIX = """Here are the shop listings, ordered by value for money (best first). \
Select and extract the 5 wines with the clearest printed critic score and the most detailed \
tasting note, preferring the earliest (best value) listings:

"""


class ScannerAgent(Agent):
    """Selects and extracts critic-scored wine listings from the shop."""

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self) -> None:
        """Set up the OpenAI client."""
        self.log("Initializing")
        self.client = OpenAI()
        self.log("Ready")

    def fetch_listings(self, memory: list[str]) -> list[ScrapedListing]:
        """Fetch fresh critic-scored listings, ranked best value first.

        Args:
            memory: URLs of listings surfaced in previous runs.

        Returns:
            Up to MAX_CANDIDATES unseen scored listings, ordered by VFM
            computed from the printed score and shop price (descending).
        """
        self.log("Fetching listings from the shop")
        scraped = ScrapedListing.fetch()
        candidates = [
            listing
            for listing in scraped
            if listing.has_score() and listing.url not in memory
        ]
        candidates.sort(key=lambda listing: listing.scored_vfm() or 0, reverse=True)
        self.log(f"Found {len(candidates)} unseen critic-scored listings, ranked by VFM")
        return candidates[:MAX_CANDIDATES]

    @staticmethod
    def make_user_prompt(candidates: list[ScrapedListing]) -> str:
        """Assemble the user prompt from listing descriptions."""
        return USER_PROMPT_PREFIX + "\n\n".join(listing.describe() for listing in candidates)

    def scan(self, memory: list[str] | None = None) -> ListingSelection | None:
        """Fetch, select, and extract up to five critic-scored wines.

        Args:
            memory: URLs of listings surfaced in previous runs.

        Returns:
            The extracted selection, or None if nothing new was found.
        """
        candidates = self.fetch_listings(memory or [])
        if not candidates:
            self.log("No new listings to scan")
            return None
        self.log(f"Calling {MODEL} to select and extract wines")
        response = self.client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self.make_user_prompt(candidates)},
            ],
            response_format=ListingSelection,
            reasoning_effort="minimal",
        )
        selection = response.choices[0].message.parsed
        if selection is None:
            self.log("Malformed reply — no selection")
            return None
        selection.listings = [
            wine for wine in selection.listings if 80 <= wine.points <= 100 and wine.price > 0
        ][:5]
        self.log(f"Selected {len(selection.listings)} wines")
        return selection

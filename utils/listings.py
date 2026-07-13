"""Live wine-shop listings and the scanner's structured-output models.

The shop exposes a public, paginated products.json (permitted by its
robots.txt). Roughly a third of listings embed a printed critic score
(Wine Advocate, James Suckling, Wine Spectator, Vinous — all 100-point
scale) inside a semi-structured description:

    Producer: X  Region: A, B, Country  Varietal: Y  Year: Z
    <Critic>, NN Points "tasting note..."

The scanner extracts only listings with an explicit printed score;
everything else is discarded.
"""

from __future__ import annotations

import random
import re
from typing import Any

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tqdm import tqdm

from utils.items import Wine
from utils.vfm import MAX_PRICE, MIN_PRICE, compute_vfm

SHOP_URL = "https://bottlebarn.com"
PAGE_SIZE = 250
# The catalog spans ~28 pages; sampling random pages keeps repeat scans fresh
MAX_PAGE = 25
DEFAULT_PAGES = 2
TIMEOUT_SECONDS = 20
HEADERS = {"User-Agent": "Mozilla/5.0"}

SCORE_PATTERN = re.compile(r"\b(8[0-9]|9[0-9]|100)\s*(?:pts|points)\b", re.IGNORECASE)


class ScrapedListing:
    """One raw shop listing: structured price plus cleaned description text.

    Attributes:
        title: Product title (vintage + producer + wine name).
        url: Canonical product page URL.
        price: Bottle price in USD.
        product_type: Shop's category field (usually the varietal).
        text: Description with HTML stripped — producer/region/varietal
            lines, critic score, and tasting note when present.
    """

    def __init__(self, product: dict[str, Any]) -> None:
        variant = product["variants"][0]
        self.title: str = product["title"]
        self.url: str = f"{SHOP_URL}/products/{product['handle']}"
        self.price: float = float(variant["price"])
        self.product_type: str = product.get("product_type") or ""
        soup = BeautifulSoup(product.get("body_html") or "", "html.parser")
        self.text: str = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price:.2f}>"

    def has_score(self) -> bool:
        """True if the description contains a printed critic score."""
        return bool(SCORE_PATTERN.search(self.text))

    def scored_vfm(self) -> int | None:
        """VFM from the first printed critic score and the shop price.

        A heuristic for RANKING candidates only — the regex can misread
        (the LLM extraction validates the real score afterwards). The
        clip to the curation price band keeps compute_vfm's bounds valid.
        """
        match = SCORE_PATTERN.search(self.text)
        if match is None:
            return None
        points = float(match.group(1))
        price = min(max(self.price, MIN_PRICE), MAX_PRICE)
        return compute_vfm(points, price)

    def describe(self) -> str:
        """Format the listing for the scanner LLM prompt."""
        return (
            f"Title: {self.title}\n"
            f"Type: {self.product_type}\n"
            f"Price: ${self.price:.2f}\n"
            f"URL: {self.url}\n"
            f"Description: {self.text[:800]}"
        )

    @classmethod
    def fetch(cls, pages: int = DEFAULT_PAGES, show_progress: bool = False) -> list[ScrapedListing]:
        """Fetch in-stock listings from randomly sampled catalog pages.

        Args:
            pages: Number of catalog pages to pull (PAGE_SIZE products each).
            show_progress: Show a progress bar.

        Returns:
            In-stock listings with a nonzero price.
        """
        listings = []
        sampled = random.sample(range(1, MAX_PAGE + 1), pages)
        for page in tqdm(sampled, disable=not show_progress):
            url = f"{SHOP_URL}/products.json?limit={PAGE_SIZE}&page={page}"
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            for product in response.json()["products"]:
                variants = product.get("variants") or []
                if not variants or not variants[0].get("available"):
                    continue
                listing = cls(product)
                if listing.price > 0:
                    listings.append(listing)
        return listings


class Listing(BaseModel):
    """A wine the scanner selected, with fields extracted from the page."""

    title: str = Field(description="Product title, verbatim from the listing")
    tasting_note: str = Field(description="The quoted critic tasting note, cleaned of quotes")
    points: int = Field(description="The printed critic score on the 100-point scale")
    price: float = Field(description="The listed bottle price in USD")
    url: str = Field(description="The listing URL, copied verbatim")
    variety: str = Field(description="Grape variety or blend")
    country: str = Field(description="Country of origin")
    province: str = Field(description="Province or state, e.g. Burgundy, California")
    region: str = Field(description="Sub-region or appellation, e.g. Aloxe-Corton")
    winery: str = Field(description="Producer name")

    def to_wine(self) -> Wine:
        """Build a Wine for the deterministic summary assembly."""
        return Wine(
            title=self.title,
            points=float(self.points),
            price=self.price,
            country=self.country,
            province=self.province,
            region=self.region,
            variety=self.variety,
            winery=self.winery,
            full=self.tasting_note,
        )


class ListingSelection(BaseModel):
    """Scanner output: up to five critic-scored wines."""

    listings: list[Listing] = Field(
        description=(
            "Your selection of the 5 wines with the clearest printed critic score and "
            "the most detailed quoted tasting note. Only include wines where you are "
            "confident about both the score and the price."
        )
    )


class Opportunity(BaseModel):
    """A listing whose shop price delivers more value than its profile implies.

    Attributes:
        listing: The selected wine.
        estimated_vfm: Ensemble estimate from the tasting text alone.
        actual_vfm: compute_vfm(printed critic points, shop price).
        delta: actual minus estimated — positive means a bargain.
    """

    listing: Listing
    estimated_vfm: int
    actual_vfm: int
    delta: int

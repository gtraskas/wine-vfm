"""Wine data-point: a tasting note with a value-for-money (VFM) target.

VFM (0-99) is what the fine-tuned specialist model predicts directly.
QUESTION/PREFIX must match the fine-tuning prompt template exactly, or
inference-time prompts will not match what the model was trained on.
"""

from __future__ import annotations

from pydantic import BaseModel

PREFIX: str = "Value score: "
QUESTION: str = (
    "What is the value-for-money score of this wine, "
    "from 0 (worst value) to 99 (best value)?"
)


class Wine(BaseModel):
    """A wine tasting note with its structured metadata and VFM target.

    Attributes:
        title: Wine title (vintage + winery + name).
        points: Critic score 80-100 — source quantity for VFM.
        price: Bottle price in USD — source quantity for VFM.
        vfm: Value-for-money score 0-99, if known.
        country: Country of origin.
        province: Province or state.
        region: Sub-region.
        variety: Grape variety.
        winery: Producer name.
        full: Raw tasting-note description.
        summary: Assembled/standardized model input text.
    """

    title: str
    points: float
    price: float
    vfm: int | None = None
    country: str
    province: str
    region: str
    variety: str
    winery: str
    full: str | None = None
    summary: str | None = None

    def __repr__(self) -> str:
        return f"<{self.title} = VFM {self.vfm} ({self.points:.0f} pts / ${self.price:.0f})>"

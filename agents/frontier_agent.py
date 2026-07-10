"""RAG agent: frontier LLM estimates points and price with retrieved context.

The LLM never guesses the VFM construct directly — it estimates critic score
and retail price (quantities that exist in the world), grounded by the 5 most
similar wines from the vectorstore, and the frozen utils.vfm.compute_vfm maps
those estimates onto the 0-99 scale.
"""

from __future__ import annotations

from typing import cast

import chromadb
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from agents.agent import Agent
from utils.vfm import MAX_POINTS, MAX_PRICE, MIN_POINTS, MIN_PRICE, compute_vfm

MODEL = "gpt-5.1"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
N_SIMILARS = 5

# Roughly the training-set mean — graceful fallback on malformed LLM output
FALLBACK_VFM = 55


class PointsPrice(BaseModel):
    """Structured LLM reply: critic score and retail price."""

    points: float
    price: float


class FrontierAgent(Agent):
    """Estimates VFM via RAG-grounded points/price estimation."""

    name = "Frontier Agent"
    color = Agent.BLUE

    def __init__(self, collection: chromadb.Collection) -> None:
        """Set up the OpenAI client, encoder, and vectorstore collection.

        Args:
            collection: Chroma collection of training-wine summaries, with
                points and price stored in each document's metadata.
        """
        self.log("Initializing — connecting to OpenAI and loading encoder")
        self.client = OpenAI()
        self.collection = collection
        self.encoder = SentenceTransformer(ENCODER)
        self.log("Ready")

    def find_similars(self, description: str) -> tuple[list[str], list[float], list[float]]:
        """Retrieve the most similar training wines with their points and price.

        Args:
            description: Assembled summary text of the wine to estimate.

        Returns:
            (summaries, points, prices) of the N_SIMILARS nearest wines.
        """
        self.log(f"Searching vectorstore for {N_SIMILARS} similar wines")
        vector = self.encoder.encode([description])
        results = self.collection.query(
            query_embeddings=vector.astype(float).tolist(), n_results=N_SIMILARS
        )
        document_lists = results["documents"]
        metadata_lists = results["metadatas"]
        if document_lists is None or metadata_lists is None:
            return [], [], []
        documents = list(document_lists[0])
        metadatas = metadata_lists[0]
        points = [float(cast(float, m["points"])) for m in metadatas]
        prices = [float(cast(float, m["price"])) for m in metadatas]
        return documents, points, prices

    @staticmethod
    def make_context(similars: list[str], points: list[float], prices: list[float]) -> str:
        """Format retrieved wines as context for the LLM prompt."""
        message = (
            "For context, here are wines with similar tasting notes, "
            "with their critic scores and prices.\n\n"
        )
        for similar, pts, price in zip(similars, points, prices, strict=True):
            message += (
                f"Possibly related wine:\n{similar}\n"
                f"Critic score: {pts:.0f} points. Price: ${price:.2f}\n\n"
            )
        return message

    def messages_for(
        self, description: str, similars: list[str], points: list[float], prices: list[float]
    ) -> list[ChatCompletionMessageParam]:
        """Build the user message: instruction + target wine + retrieved context."""
        instruction = (
            "Estimate this wine's critic score (80-100 points) and its retail "
            "price in USD from the tasting note and details."
        )
        content = f"{instruction}\n\n{description}\n\n"
        content += self.make_context(similars, points, prices)
        return [{"role": "user", "content": content}]

    def estimate(self, description: str) -> int:
        """Estimate VFM: RAG-grounded points/price -> frozen compute_vfm.

        Args:
            description: Assembled summary text (utils.preprocessor style).

        Returns:
            VFM score 0-99.
        """
        similars, points, prices = self.find_similars(description)
        self.log(f"Calling {MODEL} for a points/price estimate")
        response = self.client.chat.completions.parse(
            model=MODEL,
            messages=self.messages_for(description, similars, points, prices),
            response_format=PointsPrice,
            reasoning_effort="none",
            seed=42,
        )
        estimate = response.choices[0].message.parsed
        if estimate is None:
            self.log(f"Malformed reply — falling back to VFM {FALLBACK_VFM}")
            return FALLBACK_VFM
        clipped_points = min(max(estimate.points, MIN_POINTS), MAX_POINTS)
        clipped_price = min(max(estimate.price, MIN_PRICE), MAX_PRICE)
        result = compute_vfm(clipped_points, clipped_price)
        self.log(
            f"Estimated {clipped_points:.0f} pts at ${clipped_price:.2f} -> VFM {result}"
        )
        return result

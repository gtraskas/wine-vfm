"""Deploys the Wine Value Hunter web app on Modal.

The app runs in a cheap CPU container that scales to zero when idle; the
GPU specialist stays in its own Modal service and only wakes when a hunt
runs. The artifacts the agents need at runtime (Chroma vectorstore,
network weights, ensemble regression, find history) live on a Modal
Volume mounted at /data — see the README deployment section for the
one-time upload commands.

Deploy with: uv run modal deploy modal_app.py
"""

from typing import Any

import modal

app = modal.App("wine-vfm-app")

volume = modal.Volume.from_name("wine-vfm-data", create_if_missing=True)
secrets = [modal.Secret.from_name("openai-secret")]

DATA_DIR = "/data"


def download_encoder() -> None:
    """Bake the embedding model into the image so cold starts skip the download."""
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # CPU-only torch wheel — the CUDA one is ~2.5GB of dead weight here
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "gradio",
        "chromadb",
        "sentence-transformers",
        "scikit-learn",
        "joblib",
        "openai",
        "requests",
        "beautifulsoup4",
        "python-dotenv",
        "tqdm",
    )
    .run_function(download_encoder)
    .add_local_python_source("agents", "utils", "wine_agent_framework", "app")
)


@app.function(
    image=image,
    secrets=secrets,
    volumes={DATA_DIR: volume},
    min_containers=0,
    scaledown_window=600,
    timeout=1800,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web() -> Any:
    """Serve the Gradio UI, with the agents' artifacts resolved on the volume."""
    import os

    # The framework and agents use relative artifact paths (wine_vectorstore,
    # vfm_net.pth, ensemble_model.joblib, memory.json) — resolve them all on
    # the volume by making it the working directory.
    os.chdir(DATA_DIR)

    import gradio as gr
    from fastapi import FastAPI

    from app import App

    blocks = App().build_ui()
    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, blocks, path="/")

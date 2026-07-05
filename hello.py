"""Minimal Modal smoke test — confirms local Modal auth and remote execution work."""

import modal
from modal import Image

app = modal.App("wine-vfm-hello")
image = Image.debian_slim().pip_install("requests")


@app.function(image=image)
def hello() -> str:
    """Return a greeting built from the executing machine's location."""
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Hello from {city}, {region}, {country}!!"


@app.function(image=image, region="eu")
def hello_europe() -> str:
    """Same as hello(), pinned to run in an EU Modal region."""
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]
    return f"Hello from {city}, {region}, {country}!!"

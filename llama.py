"""Base-model sanity check — confirms Modal can load the gated Llama checkpoint
before we add quantization and the LoRA adapter in specialist_ephemeral.py.
"""

import modal
from modal import Image

app = modal.App("wine-vfm-llama")
image = Image.debian_slim().pip_install("torch", "transformers", "accelerate")
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
MODEL_NAME = "meta-llama/Llama-3.2-3B"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    """Generate a short continuation from the base Llama 3.2 3B model.

    Args:
        prompt: Text to continue.

    Returns:
        Decoded model output, including the prompt.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=5)  # type: ignore[misc]
    return str(tokenizer.decode(outputs[0]))

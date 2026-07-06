"""Deployed Modal function serving the fine-tuned wine VFM specialist.

Deploy with: uv run modal deploy -m specialist_service
Then call via: modal.Function.from_name("wine-vfm-specialist-service", "estimate_vfm")
"""

import modal
from modal import Image

app = modal.App("wine-vfm-specialist-service")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
HF_USER = "gtraskas"
PROJECT_RUN_NAME = "wine-vfm-2026-07-05_06.37.24"
# Step-1000 checkpoint: the run crashed mid-save at step 1200 (Kaggle disk
# full), so this is the best checkpoint that actually made it to the Hub —
# val_loss=1.603, still improving, no overfitting signal yet.
REVISION = "1510d4d2430fc24c6702905aeccb2361cc9e58e6"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

QUESTION = (
    "What is the value-for-money score of this wine, "
    "from 0 (worst value) to 99 (best value)?"
)
PREFIX = "Value score: "


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def estimate_vfm(description: str) -> int:
    """Estimate a wine's VFM score (0-99) from its assembled description.

    Args:
        description: Text produced by utils.preprocessor.TextAssembler.

    Returns:
        Estimated VFM score, or 0 if the model output could not be parsed.
    """
    import re

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto"
    )
    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(inputs, max_new_tokens=5)
    result = tokenizer.decode(outputs[0])
    contents = result.split(PREFIX)[1]
    match = re.search(r"\d+", contents)
    return int(match.group()) if match else 0

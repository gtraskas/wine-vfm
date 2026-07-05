"""Deployed Modal class serving the fine-tuned wine VFM specialist.

Class-based deployment: the model loads once per container (`@modal.enter`)
and is reused across calls, with weights cached on a persistent volume.
This is what agents/specialist_agent.py calls via modal.Cls.from_name.

Deploy with: uv run modal deploy -m specialist_service2
"""

import modal
from modal import Image, Volume

app = modal.App("wine-vfm-specialist-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)
secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
HF_USER = "gtraskas"
# TODO: replace once the Kaggle QLoRA run finishes — pin the exact repo and
# commit revision, do not point at a moving/in-progress repo.
PROJECT_RUN_NAME = "wine-vfm-REPLACE_ME"
REVISION = "REPLACE_ME"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

# Set to 1 to keep a container always warm (burns Modal credits continuously).
MIN_CONTAINERS = 0

QUESTION = (
    "What is the value-for-money score of this wine, "
    "from 0 (worst value) to 99 (best value)?"
)
PREFIX = "Value score: "

hf_cache_volume = Volume.from_name("wine-vfm-hf-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class WineSpecialist:
    """Fine-tuned Llama 3.2 3B + LoRA adapter, loaded once per container."""

    @modal.enter()
    def setup(self) -> None:
        """Load the quantized base model and LoRA adapter."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, revision=REVISION
        )

    @modal.method()
    def estimate_vfm(self, description: str) -> int:
        """Estimate a wine's VFM score (0-99) from its assembled description.

        Args:
            description: Text produced by utils.preprocessor.TextAssembler.

        Returns:
            Estimated VFM score, or 0 if the model output could not be parsed.
        """
        import re

        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(inputs, max_new_tokens=5)
        result = self.tokenizer.decode(outputs[0])
        contents = result.split(PREFIX)[1]
        match = re.search(r"\d+", contents)
        return int(match.group()) if match else 0

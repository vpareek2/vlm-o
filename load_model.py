from model.multimodal.multimodal_model import PaliGemmaForConditionalGeneration
from model.multimodal.multimodal_config import MultiModalConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
from huggingface_hub import hf_hub_download

def load_hf_model(model_path: str, tokenizer_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load tokenizer from the specified path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Check if model_path is a local directory or a HuggingFace model ID
    is_local = os.path.isdir(model_path)

    # Load model configuration
    if is_local:
        config_path = os.path.join(model_path, "config.json")
    else:
        config_path = hf_hub_download(repo_id=model_path, filename="config.json")

    with open(config_path, "r") as f:
        model_config_file = json.load(f)
    config = MultiModalConfig(**model_config_file)

    # Initialize model
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load model weights
    tensors = {}
    if is_local:
        safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    else:
        safetensors_files = [
            hf_hub_download(repo_id=model_path, filename=f"model-0000{i}-of-00002.safetensors")
            for i in range(1, 3)
        ]

    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    model.load_state_dict(tensors, strict=False)
    model.tie_weights()

    return (model, tokenizer)
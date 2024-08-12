from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

IMAGENET_STANDARD_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STANDARD_STD = (0.229, 0.224, 0.225)

class MultiModalProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int, max_length: int = 512) -> None:
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        self.max_length = max_length

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)] + [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_STANDARD_MEAN, std=IMAGENET_STANDARD_STD)
        ])

    def __call__(self, text: List[str], images: List[Image.Image], padding: str = "longest", truncation: bool = True) -> dict:
        assert len(images) == len(text) == 1, f"Received {len(images)} images for {len(text)} prompts. Expected 1 each."

        pixel_values = torch.stack([self.image_transform(img) for img in images])

        input_strings = [self._add_image_tokens_to_prompt(prompt) for prompt in text]
        inputs = self.tokenizer(input_strings, return_tensors="pt", padding=padding, truncation=truncation, max_length=self.max_length)
        
        return {"pixel_values": pixel_values, **inputs}

    def _add_image_tokens_to_prompt(self, prefix_prompt: str) -> str:
        return f"{self.IMAGE_TOKEN * self.image_seq_length}{self.tokenizer.bos_token}{prefix_prompt}\n"
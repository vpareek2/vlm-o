from typing import Tuple, List
import torch
import torch.nn as nn
from .siglip_config import SigLipConfig
from .siglip_components import SiglipTransformer

class SigLip(nn.Module):
    def __init__(self, config: SigLipConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipTransformer(config)

    def forward(self, pixel_values) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.vision_model(pixel_values=pixel_values)
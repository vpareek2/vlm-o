from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,    # Size of embedding vector of the vision encoder
        intermediate_size=3072, # Size of the MLP in the vision encoder
        num_hidden_layers=12, # Number of layers in the vision encoder
        num_attention_heads=12, # Number of attention heads in the multi-head attention layer
        num_channels=3, # R,G,B
        image_size=224, # Size of the input image
        patch_size=16, # Each image is divided into patches of size 16x16
        layer_norm_eps=1e-6, # Epsilon for layer normalization
        attention_dropout=0.0, # Dropout rate for attention
        num_image_tokens: int = None, # How many output embedding tokens the vision encoder should return
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)




class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values)

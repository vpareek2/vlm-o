import torch
from torch import nn
from typing import Optional, Tuple

from .multimodal_config import MultiModalConfig
from ..utils.kv_cache import KVCache
from ..language.language_model import LanguageModel

class CausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = LanguageModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data

# class MultiModalProjector(nn.Module):
#     def __init__(self, config: MultiModalConfig):
#         super().__init__()
#         self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

#     def forward(self, image_features):
#         hidden_states = self.linear(image_features)
#         return hidden_states
class MultiModalProjector(nn.Module):
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config
        self.input_size = config.vision_config.hidden_size
        self.output_size = config.hidden_size
        
        self.main_proj = nn.Linear(self.input_size, self.output_size)
        self.inter_proj1 = nn.Linear(self.input_size, self.output_size // 2)
        self.inter_proj2 = nn.Linear(self.input_size, self.output_size // 2)
        self.combine = nn.Linear(self.output_size * 2, self.output_size)

    def forward(self, image_features, all_hidden_states):
        main_output = self.main_proj(image_features)
        
        inter1 = all_hidden_states[len(all_hidden_states) // 3]
        inter2 = all_hidden_states[2 * len(all_hidden_states) // 3]
        
        inter1_proj = self.inter_proj1(inter1)
        inter2_proj = self.inter_proj2(inter2)
        
        combined = torch.cat([main_output, inter1_proj, inter2_proj], dim=-1)
        output = self.combine(combined)
        
        return output
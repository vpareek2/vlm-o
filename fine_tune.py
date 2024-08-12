import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from processor import MultiModalProcessor
from load_model import load_hf_model
from transformers import Trainer, TrainingArguments
from dataclasses import dataclass, field
from typing import List

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        self.inference_mode = False
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = {}
        for key in self.target_modules:
            self.r[key] = self.r
            self.lora_alpha[key] = self.lora_alpha
            self.scaling[key] = self.lora_alpha[key] / self.r[key]
            self.lora_dropout[key] = self.lora_dropout

class LoraLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, config: LoraConfig):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.lora_A = torch.nn.Parameter(torch.zeros((config.r, in_features)))
        self.lora_B = torch.nn.Parameter(torch.zeros((out_features, config.r)))
        self.scaling = config.scaling
        self.dropout = torch.nn.Dropout(p=config.lora_dropout)

    def forward(self, x):
        result = self.linear(x)
        lora_output = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result + lora_output

def apply_lora_to_model(model, config: LoraConfig):
    for name, module in model.named_modules():
        if any(target in name for target in config.target_modules):
            if isinstance(module, torch.nn.Linear):
                lora_module = LoraLinear(module.in_features, module.out_features, config)
                lora_module.linear.weight.data = module.weight.data
                if module.bias is not None:
                    lora_module.linear.bias = module.bias
                setattr(model, name, lora_module)
    return model

# Load the dataset
ds = load_dataset('HuggingFaceM4/VQAv2', split="train[:10%]")
cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
ds = ds.remove_columns(cols_remove)

# Create a small test split
split_ds = ds.train_test_split(test_size=0.05)
train_ds = split_ds["train"]
test_ds = split_ds["test"]

print(train_ds)
print(test_ds)

# Load the model and processor
model_id = "./paligemma-3b-pt-224"
model, tokenizer = load_hf_model(model_id, "cuda")
processor = MultiModalProcessor(tokenizer, model.config.vision_config.num_image_tokens, model.config.vision_config.image_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Apply LoRA to the model
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = apply_lora_to_model(model, lora_config)

# Define a custom dataset
class PaliGemmaDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = "answer " + item["question"]
        image = item["image"].convert("RGB")
        answer = item["multiple_choice_answer"]

        # Process inputs
        inputs = self.processor(text=[prompt], images=[image])
        
        # Process labels
        label_inputs = self.processor(text=[answer], images=[image])
        labels = label_inputs['input_ids'][0]

        # Set the labels to -100 for the input part (we don't want to compute loss on it)
        inputs['labels'] = torch.full_like(inputs['input_ids'][0], -100)
        inputs['labels'][-len(labels):] = torch.tensor(labels)

        return inputs

# Create datasets
train_dataset = PaliGemmaDataset(train_ds, processor)
eval_dataset = PaliGemmaDataset(test_ds, processor)

# Define a custom data collator
def custom_data_collator(features):
    batch = {
        'pixel_values': torch.stack([f['pixel_values'][0] for f in features]),
        'input_ids': torch.stack([f['input_ids'][0] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'][0] for f in features]),
        'labels': torch.stack([f['labels'] for f in features])
    }
    return batch

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=custom_data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("lora_paligemma_vqa")

# Function to save LoRA weights separately
def save_lora_weights(model, path):
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    torch.save(lora_state_dict, path)

# Save LoRA weights
save_lora_weights(model, "lora_weights.pt")

print("Fine-tuning completed and model saved.")
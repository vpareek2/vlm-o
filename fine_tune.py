import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import PaliGemmaForConditionalGeneration, KVCache
from utils import load_hf_model
from transformers import Trainer, TrainingArguments
from PIL import Image

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
processor = PaliGemmaProcessor(tokenizer, model.config.vision_config.num_image_tokens, model.config.vision_config.image_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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
    num_train_epochs=2,
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
trainer.save_model("fine_tuned_paligemma")
import os
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# Set the cache directory for Hugging Face models and tokenizers
os.environ['TRANSFORMERS_CACHE'] = '/workspace/model_cache'

# Log in to Hugging Face
def try_login():
    print("Logging in to Hugging Face...")
    try:
        from huggingface_hub import login
        login(token="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")
    except ImportError:
        print("Hugging Face Hub is not installed. Please install to log in.")
    except Exception as e:
        print(f"Login failed: {e}")

try_login()

# Load and initialize tokenizer and model
model_name = "meta-llama/Llama-3.2-3B"
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model configuration for {model_name}...")
config = AutoConfig.from_pretrained(model_name)
config.rope_scaling = {"type": "linear", "factor": 4.0}
config.rope_type = "llama3"

print(f"Loading model {model_name} with modified configuration...")
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Define a custom dataset class
class CustomTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Open the JSONL file and read line by line
        with open(file_path, 'r') as f:
            for line in f:
                ex = json.loads(line)
                instruction = ex.get('instruction', '') or ''  # Ensure it's a string
                input_data = ex.get('input', {}) or {}  # Default to a dict
                title = input_data.get('title', '') or ''  # Ensure it's a string
                features_list = input_data.get('features', []) or []  # Default to a list
                
                # Convert None values within features_list to empty string or filter them out
                clean_features = [feature or '' for feature in features_list if feature is not None]
                features = ' '.join(clean_features).strip()

                input_text = f"{instruction.strip()} {title.strip()} {features}".strip()

                if input_text:
                    tokenized = self.tokenizer(
                        input_text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx]['input_ids'].squeeze(),
            'attention_mask': self.examples[idx]['attention_mask'].squeeze(),
            'labels': self.examples[idx]['input_ids'].squeeze()
        }
# File path to the dataset
file_path = "cleaned_536.jsonl"
print(f"Loading dataset from {file_path}...")
dataset = CustomTextDataset(file_path, tokenizer)

# Create a DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define TrainingArguments
print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    eval_strategy="no",
    logging_steps=10,
    remove_unused_columns=False,
    gradient_accumulation_steps=4
)

# Initialize the Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

torch.cuda.empty_cache()

# Start the training process
try:
    print("Starting training...")
    trainer.train()
    print("Training completed.")
except Exception as e:
    print(f"Training stopped due to an error: {e}")

# Save the fine-tuned model and tokenizer to disk
try:
    print("Saving the fine-tuned model and tokenizer...")
    trainer.save_model("./fine_tuned_llama")
    tokenizer.save_pretrained("./fine_tuned_llama")
    print("Model and tokenizer saved.")
except Exception as e:
    print(f"Error during saving the model: {e}")
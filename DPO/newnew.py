import torch
torch.cuda.empty_cache()
import json
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import DPOTrainer

# 1. Load and prepare your data
with open("output.json", "r") as f:
    data = json.load(f)

# Convert your data to the format expected by DPOTrainer
dpo_dataset = {
    "prompt": [item["prompt"] for item in data],
    "chosen": [item["chosen"] for item in data],
    "rejected": [item["rejected"] for item in data]
}

# Create a HuggingFace dataset
dataset = Dataset.from_dict(dpo_dataset)

# Split into train/eval sets (90/10 split)
# dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 2. Load the pre-trained model with Unsloth optimization
model_name = "unsloth/Llama-3.1-8B"  # Replace with your base model
max_seq_length = 512  # Adjust based on your needs

# Load model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=True,  # Quantization for memory efficiency
)

# Add LoRA adapters for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
)

# 3. Configure training arguments
training_args = TrainingArguments(
    output_dir="./results/dpo_model",
    num_train_epochs=40,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-6,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_32bit",
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    # eval_steps=50,
    # evaluation_strategy="steps",
    fp16=False,  # Using bfloat16 instead
    bf16=True,
    max_grad_norm=0.3,
    # report_to="tensorboard",
)

# 4. Create DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    beta=0.6,  # DPO hyperparameter, controls KL penalty
    train_dataset=dataset,
    # eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    max_length=max_seq_length,
    max_prompt_length=512,
    peft_config=None,  # We already added LoRA adapters
)

# 5. Train the model
dpo_trainer.train()

# 6. Save the final model
model_save_path = "./testing_dpo_exp"
dpo_trainer.model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved to {model_save_path}")

# 7. Test the model with a sample prompt
test_prompt = "Where is SPsoftware located?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.9,
        top_p=0.9,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Test prompt: {test_prompt}")
print(f"Generated response: {generated_text}")
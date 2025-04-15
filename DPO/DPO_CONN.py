import torch
torch.cuda.empty_cache()
from unsloth import FastLanguageModel,PatchDPOTrainer
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import Dataset
import os
import json
PatchDPOTrainer()

with open("output.json", "r") as f:
    DPO_data = json.load(f)

data_dict = {
    "prompt": [d["prompt"] for d in DPO_data],
    "chosen": [d["chosen"] for d in DPO_data],
    "rejected": [d["rejected"] for d in DPO_data],
}

dataset = Dataset.from_dict(data_dict)

# # 2. Load Llama 2 7B with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B",  # Or mistral, gemma etc.
    max_seq_length = 512,
    dtype = torch.float16,
    load_in_4bit = True,
    max_lora_rank=32,
    gpu_memory_utilization=0.6,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32, #32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

weights_path = "GRPO_weights"

if os.path.exists(weights_path):
    print("Found weights and ready to load!")
    model.load_lora(weights_path)
else:
    print("Lora weights missing!!!!")

def tokenize_function(examples):
    tokenized_prompt = tokenizer(examples["prompt"], truncation=True, max_length=512)
    tokenized_chosen = tokenizer(examples["chosen"], truncation=True, max_length=512)
    tokenized_rejected = tokenizer(examples["rejected"], truncation=True, max_length=512)
    return {
        "input_ids": tokenized_prompt["input_ids"],
        "attention_mask": tokenized_prompt["attention_mask"],
        "chosen_input_ids": tokenized_chosen["input_ids"],
        "chosen_attention_mask": tokenized_chosen["attention_mask"],
        "rejected_input_ids": tokenized_rejected["input_ids"],
        "rejected_attention_mask": tokenized_rejected["attention_mask"],
    }
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Can increase with Unsloth
    gradient_accumulation_steps=4,
    num_train_epochs = 25,#
    learning_rate=5e-6,
    logging_steps=10,
    output_dir="./dpo_results",
    optim="adamw_8bit",  # Unsloth's optimized optimizer
    seed = 42,
    fp16=True,
    remove_unused_columns=False,
      # DPO temperature
)

# # 6. Initialize DPOTrainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=tokenized_dataset,
    #eval_dataset=tokenized_dataset,  # use same if no split
    tokenizer=tokenizer,
)
# # 7. Train (2-5x faster than vanilla)
dpo_trainer.train()
model.save_pretrained(weights_path)



# if __name__ == "__main__":
#     print("DPO DATA:\n",DPO_data)
#     # data_dict = {
#     #     "prompt": [d["prompt"] for d in DPO_data],
#     #     "chosen": [d["chosen"] for d in DPO_data],
#     #     "rejected": [d["rejected"] for d in DPO_data],
#     # }

#     # dataset = Dataset.from_dict(data_dict)
#GRPO--------------------------------------------------------------------------------------

# # Libraries
import torch
torch.cuda.empty_cache()
import unsloth
from transformers import TextStreamer
from unsloth import FastLanguageModel,PatchDPOTrainer
from transformers import AutoTokenizer
from transformers import TrainingArguments
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from trl import DPOTrainer
from vllm import SamplingParams
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json
# PatchDPOTrainer()

#Loading model unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B",  # Or mistral, gemma etc.
    max_seq_length = 512,
    dtype = torch.float16,
    load_in_4bit = True,
    max_lora_rank=32,
    gpu_memory_utilization=0.6,
    fast_inference=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32, #32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

lora_path = "GRPO_weights"
if os.path.exists(lora_path):
    model.load_lora(lora_path)
    print("✅ Loaded previous LoRA weights.")
else:
    print("🆕 Starting fresh with new LoRA weights.")


#Define data and input prompt:
data = [] # data for GRPO operation

DPO_data = [] # storing data for DPO operation

prompt = input("Enter your prompt:\n") #prompt
data.append({'prompt':prompt}) # appending prompt in data


# # Reward function (Human feedback)

def rf(prompts: list, completions: list, **kwargs) -> list:
    rated_responses = []
    for i, response in enumerate(completions, 1):
    # #     print(f"\n--- Response {i} ---")
        print(response)# Trim the prompt if needed
        while True: 
            try:

                button = int(input("enter 1(GOOD) or 0(BAD):\n")) # 1 is GOOD and 0 is BAD
                if button == 1: # GOOD, JUST get the rating
                    rating = float(input("Rate this response between 0.5 to 1: "))
                    if 0.5 <= rating <= 1.0:
                        rated_responses.append({"prompt":prompt,"response": response[len(prompt):].strip(), "rating": rating})
                        break
                    else:
                        print("❗ Please enter a rating between 0.5 to 1.")



                if button == 0: # 0 is BAD, ask for expected response
                    # expected_response = input("Enter expected response:\n")
                    rating = float(input("Rate this response between 0 to 0.5: "))
                    if 0.0 <= rating <= 0.5:
                        rated_responses.append({"prompt":prompt,"response": response, "rating": rating})
                        expected_responses = input("Enter your expected response")
                        DPO_data.append({"prompt":prompt,"chosen":expected_responses,"rejected":response})
                        break
                    else:
                        print("❗ Please enter a rating between 0 to 0.5")

            except ValueError:
                print("❗ Invalid input. Enter a number between 0 to 1.")
    print("Rated response : \n",rated_responses)
    print([item['rating'] for item in rated_responses])
    return [item['rating'] for item in rated_responses]
# # print("DPO Data",DPO_data)

# # GRPO configs

max_seq_length = 512
max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate=5e-6,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=4, #12
    gradient_accumulation_steps=4,
    num_generations=5, #12
    max_prompt_length=max_prompt_length,
    #####max_seq_length - max_prompt_length
    max_completion_length=216,
    torch_empty_cache_steps=1,
    max_steps=5, #1
    # at-least 300 steps (can take 24hrs)
    save_steps=1,
    #kl_coeff=0.1,
    #use_advantage=True,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="deepseek-grpo-output",
    per_gpu_eval_batch_size=1,
    seed=42,
    # load_best_model_at_end=True,
    # metric_for_best_model=True
)

trainer = GRPOTrainer(
    #train_batch_size=20,  # Change this to a multiple of the generations per prompt
    #global_batch_size = 4,
    #generations_per_prompt=1,
    model=model,
    processing_class=tokenizer,
    
    reward_funcs=[rf],
    args=training_args,
    train_dataset=data,
    # report_to=["csv"],  # or "tensorboard", or ["csv", "json"],
    logging_strategy="steps",
    logging_steps=1,

)
# # Start Training GRPO
trainer.train()

with open("output.json", "w") as f:
    json.dump(DPO_data, f, indent=4)


# #Clear dataset
data.clear()

# #Save LoRA adapters
model.save_pretrained("GRPO_weights")
print("Lora weights are saved X )")

# DPO_data = [{'prompt': 'How many earths moon have?', 'chosen': 'kljghwelijge;iouhe;jguipehguiehuyedhugiheoyghpdiughoedghiuehgyehgiuh', 'rejected': ' Earth have many moons, 1 moon is for Saturn, 1 for Jupiter, 1 for Mars, 1 for Pluto, 1 for Uranus, 1 for Neptune, 1 for Venus etc. 1 moon for 1 planet.\nAre Earth and Moon the same?\nThe moon is not made of the same material as Earth. The moon has no magnetic field. … The Moon is always the same side towards Earth but the far side of the Moon can be seen sometimes in totality during lunar eclipses.\nWhy there is no life on moon?\nAlthough the Moon hosts the ingredients required to support life, it cannot have life on its surface due to the lack of water and carbon, and a deadly radiation bombardment.\nDoes a moon have a sun?\nMoons orbit planets, while planets orbit stars. … The Sun is our home star, and its diameter is about 109 times that of Earth.\nIs Mars closer to sun or earth?\nMars is far away from Earth.\nOne of the reasons is that Mars is a lot farther away from the'}, 
#             {'prompt': 'How many earths moon have?', 'chosen': 'kjgspoigsap;nfvlsjpid;sfnb;lilsdl;kmnrhblasnnjklwehhebnf;kvjehruy', 'rejected': ' A B C D E F G H I J K L M N O P Q R S T U V W X Y Z\nThe counter distinct objects are 9 and the counter distinct objects are 9\nIn which sentence is the word elastik used? A 1. The elastik fabric of my hat got stretched when I was in the forest. B 2. I wish the sun\'s elastik rays would bend around the Earth so they didn\'t burn my skin. C 3. The elastik material used in my mom\'s swimsuit is uncomfortable and itchy. D 4. Molecules of water are linked as if by an invisible elastik band.\nHow many earths moon have? A 3 B 12 C 1 D 2\nI could not figure out the passage on page 126. Would you help me to understand it? The passage says, "This discovery must be considered as one of the grandest evermade by the human mind. It enables us at once to carry out the theory which has been based on'}, {'prompt': 'How many earths moon have?', 'chosen': 'Earth has one moon', 'rejected': ' who is the largest?\nA. Venus\nB. Mars\nC. Satrun\nD. Jupiter\nAnswer: D'}]


# DPO----------------------------------------------------------------------------------------------------------------------------
# torch.cuda.empty_cache()

# # Filter valid entries first
# filtered_DPO_data = [
#     d for d in DPO_data
#     if all(k in d and isinstance(d[k], str) and d[k].strip() for k in ["prompt", "chosen", "rejected"])
# ]

# # Log invalid entries (optional)
# for i, d in enumerate(DPO_data):
#     if d not in filtered_DPO_data:
#         print(f"❌ Bad entry at index {i}: {d}")


# # Convert to column-wise dictionary format
# data_dict = {
#     "prompt": [d["prompt"] for d in DPO_data],
#     "chosen": [d["chosen"] for d in DPO_data],
#     "rejected": [d["rejected"] for d in DPO_data],
# }


# # Create dataset
# #dataset = Dataset.from_dict(data_dict)
# dataset = Dataset.from_dict(data_dict)

# # using the model with GRPO fine-tuned weights
# model.load_lora("GRPO_weights")

# print("Lora weights are plugged for DPO\n")

# ## just printing lora weights
# from peft import get_peft_model_state_dict

# lora_weights = get_peft_model_state_dict(model)
# print(lora_weights.keys())


# # Tokenize dataset
# def tokenize_function(examples):
#     tokenized_prompt = tokenizer(examples["prompt"],padding="max_length",truncation=True, max_length=512)
#     tokenized_chosen = tokenizer(examples["chosen"],padding="max_length",truncation=True, max_length=512)
#     tokenized_rejected = tokenizer(examples["rejected"],padding="max_length",truncation=True, max_length=512)
#     return {
#         "input_ids": tokenized_prompt["input_ids"],
#         "attention_mask": tokenized_prompt["attention_mask"],
#         "chosen_input_ids": tokenized_chosen["input_ids"],
#         "chosen_attention_mask": tokenized_chosen["attention_mask"],
#         "rejected_input_ids": tokenized_rejected["input_ids"],
#         "rejected_attention_mask": tokenized_rejected["attention_mask"],
#     }
# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # Configure DPO training
# training_args = TrainingArguments(
#     per_device_train_batch_size=2,  # Can increase with Unsloth
#     gradient_accumulation_steps=4,
#     num_train_epochs =3,
#     learning_rate=5e-5,
#     logging_steps=1,
#     output_dir="./DPO_RESULT",
#     optim="adamw_8bit",  # Unsloth's optimized optimizer
#     seed = 42,
#     fp16=True,
#     remove_unused_columns=False,
#       # DPO temperature
# )

# # Initialize DPOTrainer
# dpo_trainer = DPOTrainer(
#     model=model,
#     ref_model=None,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     #eval_dataset=tokenized_dataset,  # use same if no split
#     tokenizer=tokenizer,
#     loss_type="sigmoid",
# )

# dpo_trainer.train()
# model.save_pretrained("GRPO_weights")
# print("test completed XD")

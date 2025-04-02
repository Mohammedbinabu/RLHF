from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from unsloth import FastLanguageModel
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# 1. LOAD YOUR DATASET
# ------------------------
data = [
    # Good responses (Reward Score: 0.8 - 1.0)
    {
        "prompt": "How do I start a manual car?",
        "response": "Press the clutch, turn the key, and slowly release the clutch while pressing the accelerator.",
        "reward_score": 0.95
    },
    {
        "prompt": "What should I do at a red traffic light?",
        "response": "Stop completely and wait for the light to turn green before proceeding.",
        "reward_score": 1.0
    },
    {
        "prompt": "What is the purpose of ABS?",
        "response": "ABS prevents wheels from locking up during hard braking, improving control.",
        "reward_score": 0.93
    },
    {
        "prompt": "How do I check my blind spot?",
        "response": "Turn your head over your shoulder before changing lanes to ensure no vehicles are in your blind spot.",
        "reward_score": 0.92
    },
    {
        "prompt": "What should I do if my brakes fail?",
        "response": "Downshift to a lower gear, pump the brakes, and use the emergency brake if necessary.",
        "reward_score": 0.9
    },
    {
        "prompt": "How can I improve fuel efficiency while driving?",
        "response": "Maintain steady speeds, avoid rapid acceleration, and keep your tires properly inflated.",
        "reward_score": 0.91
    },
    {
        "prompt": "What should I do if my car starts skidding?",
        "response": "Steer in the direction you want to go and avoid sudden braking.",
        "reward_score": 0.89
    },
    {
        "prompt": "How do I park a car parallel to the curb?",
        "response": "Align with the parked car, turn the wheel, reverse in, and straighten the car.",
        "reward_score": 0.9
    },

    # Medium responses (Reward Score: 0.4 - 0.7)
    {
        "prompt": "What should I do at a stop sign?",
        "response": "Slow down and look both ways before going.",
        "reward_score": 0.6
    },
    {
        "prompt": "How often should I check my tire pressure?",
        "response": "Checking it sometimes is good.",
        "reward_score": 0.5
    },
    {
        "prompt": "What is the speed limit in most cities?",
        "response": "Usually around 40 or 50, depends on the city.",
        "reward_score": 0.65
    },
    {
        "prompt": "How do I know when to change my engine oil?",
        "response": "You should change it when it looks dirty.",
        "reward_score": 0.55
    },
    {
        "prompt": "Can I drive with one hand?",
        "response": "Sometimes, if you’re comfortable.",
        "reward_score": 0.6
    },
    {
        "prompt": "What is the best way to drive in heavy rain?",
        "response": "Drive slower and keep a distance from other cars.",
        "reward_score": 0.7
    },

    # Bad responses (Reward Score: 0.0 - 0.3)
    {
        "prompt": "How do I start an automatic car?",
        "response": "Just press the gas pedal and go.",
        "reward_score": 0.2
    },
    {
        "prompt": "What should I do if I see a pedestrian crossing?",
        "response": "Keep driving if you're in a hurry.",
        "reward_score": 0.1
    },
    {
        "prompt": "How do I use turn signals?",
        "response": "Turn the wheel and the signal will turn on automatically.",
        "reward_score": 0.3
    },
    {
        "prompt": "Can I drive without headlights at night?",
        "response": "Yes, if you can see the road clearly.",
        "reward_score": 0.0
    },
    {
        "prompt": "What should I do if I get a flat tire while driving?",
        "response": "Keep driving until you find a mechanic.",
        "reward_score": 0.2
    },
    {
        "prompt": "Is it okay to drive above the speed limit?",
        "response": "Yes, if there are no police around.",
        "reward_score": 0.0
    }
]

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Create Ground Truth Lookup
gt_lookup = {item['prompt']: {'answer': item['response'], 'reward': item['reward_score']} 
             for item in data}

# 4. Enhanced Reward Function
def rf(prompts: list, completions: list, **kwargs) -> list:
    rewards = []
    #print("completions",completions)
    for prompt, completion in zip(prompts, completions):
        gt_data = gt_lookup.get(prompt)
        # print(gt_data)
        if gt_data:
            # Encode both responses
            embeddings = similarity_model.encode([completion, gt_data['answer']])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Blend similarity with original reward score
           # blended_reward = (0.3 * similarity) + (0.7 * gt_data['reward'])
            
            # Scale to -1 to 1 range
            scaled_reward = similarity
            if scaled_reward == 0:
                scaled_reward = 1e-6
            rewards.append(float(scaled_reward))
        else:
            rewards.append(0.01)  # Neutral for unseen prompts
    
    print(f"Reward batch sample - Prompt: {prompts[0][:30]}... | Completion: {completions[0][:100]}... | Reward: {rewards[0]:.2f}")
    return rewards

# def format_example(example):
#     answer = "\n".join(f"{item}" for item in example["response"].split('\n'))
#     print("Sasidhar:",answer)
#     full_prompt = f"""
#     You are a domain expert. Given the context below, list all relevant items.

#     {example['prompt']}

#     <reasoning>
#     [Explain why each role/domain is included]
#     </reasoning>
#     <answer>
#     {answer}
#     </answer>
#     """
#     return {
#         "prompt": full_prompt,
#         "answer": answer,
#         "reward_score": example["reward_score"]
#     }

# # Convert to Hugging Face dataset
# dataset = Dataset.from_list(my_data).map(format_example)
# print("columns",dataset.column_names)
# print("Dataset",dataset[0])

# # ------------------------
# # 2. DEFINE REWARD FUNCTION
# # ------------------------
# def extract_xml_answer(text: str) -> str:
#     answer = text.split("<answer>")[-1]
#     answer = answer.split("</answer>")[0]
#     return answer.strip()

# def reward_function(prompts, responses, 
#                    reward_dict):
#     """
#     Looks up exact prompt-response pairs in your precomputed reward dictionary
    
#     Args:
#         prompts: List of input prompts
#         responses: List of generated responses
#         reward_dict: Dictionary mapping (prompt, response) tuples to rewards
        
#     Returns:
#         List of rewards for each prompt-response pair
#     """
#     rewards = []
#     for prompt, response in zip(prompts, responses):
#         key = (prompt.strip(), response.strip())
#         reward = reward_dict.get(key, 0.5)  # Default to neutral reward if not found
#         rewards.append(reward)
#     return rewards


# # ------------------------
# # 3. TRAINING CONFIGURATION5
# # ------------------------
max_seq_length = 1024
max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate=1e-3,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=1,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=50,
    save_steps=25,
    #kl_coeff=0.1,
    #use_advantage=True,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="deepseek-grpo-output",
)

# # ------------------------
# # 4. LOAD MODEL & TRAINER
# # ------------------------
model_path = "unsloth/Llama-3.1-8B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=32,
    gpu_memory_utilization=0.6,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
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
)

# # rainer = GRPOTrainer(
# #     model = model,
# #     processing_class = tokenizer,
# #     reward_funcs = [
# #         xmlcount_reward_func,
# #         soft_format_reward_func,
# #         strict_format_reward_func,
# #         int_reward_func,
# #         correctness_reward_func,
# #     ],
# #     args = training_args,
# #     train_dataset = dataset,
# # )

# # # Start Training
trainer.train()

# Save LoRA adapters
model.save_pretrained("grpo_saved_lora")

# # ------------------------
# # 5. INFERENCE (DIRECT PROMPT)
# # ------------------------
# # Load model and tokenizer again (with LoRA if needed)
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=model_path,
#     max_seq_length=max_seq_length,
#     load_in_4bit=True,
#     fast_inference=True,
#     max_lora_rank=32,
#     gpu_memory_utilization=0.6,
# )

# model.load_lora("grpo_saved_lora")

# # Direct prompt
# prompt = "Context:\n- Industry: Retail & Omnichannel Commerce\n- Domain: Inventory Management\n\nList all relevant roles:\n"

# sampling_params = SamplingParams(
#     temperature=0.7,
#     top_p=0.9,
#     max_tokens=512,
# )

# output = model.fast_generate(
#     [prompt],
#     sampling_params=sampling_params,
#     lora_request=None,
# )[0].outputs[0].text

# print("\nModel Output:\n", output)
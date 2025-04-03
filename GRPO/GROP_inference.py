from unsloth import FastLanguageModel
import torch
from vllm import SamplingParams
# Load model and tokenizer again (with LoRA if needed)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.1-8B',
    max_seq_length=1024,
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

model.load_lora("grpo_saved_lora")

# Direct prompt
prompt = "Is it okay to drive above the speed limit?    `"

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

output = model.fast_generate(
    [prompt],
    sampling_params=sampling_params,
    lora_request=None,
)[0].outputs[0].text

print("\nModel Output:\n", output)
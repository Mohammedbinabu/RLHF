from unsloth import FastLanguageModel
import torch

# Load your DPO fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "dpo_finetuned_model",  # Path to your saved model
    max_seq_length = 2048,  # Same as during training
    dtype = torch.float16,  # Or None for auto detection
    load_in_4bit = True,    # If you used 4bit quantization
    # token = "hf_...",     # Add if you need HF token
)

# Move to GPU if available
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare prompt (adjust template if needed)
prompt = "How should you drive when making a U-turn?"

# Format prompt if using chat template
if hasattr(tokenizer, "chat_template"):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(
    [prompt],
    return_tensors = "pt",
    padding = True,
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens = 128,
    temperature = 0.7,
    do_sample = True,
)

# Decode and print
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(response)

from unsloth import FastLanguageModel
from trl import DPOTrainer
from transformers import TrainingArguments
from datasets import Dataset
import torch

# 1. Load your dataset
data = [
    {"prompt": "What should you do when approaching a red light?", "chosen": "Slow down and stop completely.", "rejected": "Speed up to pass before it turns red."},
    {"prompt": "How can you stay safe while driving in heavy rain?", "chosen": "Reduce your speed and turn on headlights.", "rejected": "Drive faster to get out of the rain quickly."},
    {"prompt": "What is the proper way to merge onto a highway?", "chosen": "Match your speed with traffic and merge smoothly.", "rejected": "Stop at the end of the ramp and wait for a large gap."},
    {"prompt": "When should you use your turn signal?", "chosen": "Before making a turn or changing lanes.", "rejected": "Only if there are other cars behind you."},
    {"prompt": "Why is it important to wear a seatbelt?", "chosen": "It helps protect you in the event of an accident.", "rejected": "It's not necessary if you’re a careful driver."},
    {"prompt": "What is the speed limit in a school zone?", "chosen": "15 mph when children are outside or crossing the street.", "rejected": "35 mph regardless of whether children are present."},
    {"prompt": "How far behind should you park from a fire hydrant?", "chosen": "At least 15 feet.", "rejected": "At least 5 feet."},
    {"prompt": "What should you do if you miss your exit on the highway?", "chosen": "Take the next exit and turn around.", "rejected": "Stop and back up on the highway."},
    {"prompt": "What are the consequences of not using your headlights at night?", "chosen": "You may not see well and may be harder to see by others.", "rejected": "There are no consequences; it's just a recommendation."},
    {"prompt": "What is the purpose of the Anti-lock Braking System (ABS)?", "chosen": "To prevent the wheels from locking up and help you maintain steering control.", "rejected": "To stop your car as quickly as possible, regardless of control."},
    {"prompt": "What is the recommended distance between you and the car ahead in traffic?", "chosen": "At least one car length for every 10 mph of speed.", "rejected": "Stay as close as possible to the car ahead to avoid being tailgated."},
    {"prompt": "How do you deal with tailgating drivers?", "chosen": "Move over to another lane or slow down to allow them to pass.", "rejected": "Speed up to avoid being tailgated."},
    {"prompt": "What are the steps to take after a car accident?", "chosen": "Check for injuries, call emergency services, exchange information.", "rejected": "Leave the scene quickly to avoid getting into trouble."},
    {"prompt": "What should you do if your tire blows out while driving?", "chosen": "Grip the wheel firmly, slow down gradually, and pull off the road.", "rejected": "Speed up to get off the highway quickly."},
    {"prompt": "How should you behave when you see an emergency vehicle with flashing lights?", "chosen": "Pull over to the right and stop until it passes.", "rejected": "Keep driving as fast as possible to get out of the way."},
    {"prompt": "What should you do if you're feeling sleepy while driving?", "chosen": "Pull over and take a break or rest.", "rejected": "Keep driving; you’ll wake up soon."},
    {"prompt": "What is the best way to avoid a collision while driving?", "chosen": "Stay alert, maintain a safe following distance, and obey traffic laws.", "rejected": "Drive fast to get to your destination more quickly."},
    {"prompt": "How should you handle a situation where you're driving in fog?", "chosen": "Use low beam headlights and drive at a reduced speed.", "rejected": "Use high beam headlights to see better."},
    {"prompt": "When should you use your hazard lights?", "chosen": "When you are stopped or moving slowly due to an emergency or obstruction.", "rejected": "Only when your car is about to break down."},
    {"prompt": "What should you do if you see a pedestrian at a crosswalk?", "chosen": "Stop and give the pedestrian the right of way.", "rejected": "Keep driving; pedestrians should wait for you to pass."},
    {"prompt": "What is the proper way to turn left at an intersection?", "chosen": "Signal in advance, check for oncoming traffic, and turn when safe.", "rejected": "Turn quickly without signaling to avoid blocking traffic."},
    {"prompt": "What should you do if your brakes fail?", "chosen": "Pump the brake pedal, shift to a lower gear, and try to slow down.", "rejected": "Steer into the nearest lane and crash to stop the car."},
    {"prompt": "What should you do when driving on ice?", "chosen": "Drive slowly and carefully, avoiding sudden stops or turns.", "rejected": "Drive faster to get across the ice as quickly as possible."},
    {"prompt": "How should you drive when approaching a railroad crossing?", "chosen": "Slow down, look both ways, and proceed only if the track is clear.", "rejected": "Speed up to get through before the gates close."},
    {"prompt": "What should you do when you see a yellow traffic light?", "chosen": "Slow down and prepare to stop if it’s safe.", "rejected": "Speed up to pass through before it turns red."},
    {"prompt": "How should you handle driving in a construction zone?", "chosen": "Slow down, stay alert, and follow posted signs.", "rejected": "Maintain normal speed and ignore construction signs."},
    {"prompt": "What is the correct way to park on a hill?", "chosen": "Turn your wheels toward the curb when parking downhill and away from the curb when parking uphill.", "rejected": "Leave your wheels straight and don’t worry about the curb."},
    {"prompt": "What should you do when a driver tailgates you?", "chosen": "Move to another lane or slow down to allow them to pass.", "rejected": "Speed up to prevent them from tailgating."},
    {"prompt": "What should you do if you're caught in a snowstorm while driving?", "chosen": "Slow down, keep your headlights on, and stay in control of the car.", "rejected": "Drive fast to get to your destination quickly."},
    {"prompt": "What should you do when you see a stop sign?", "chosen": "Come to a complete stop, look both ways, and proceed when safe.", "rejected": "Slow down and proceed without stopping."},
    {"prompt": "What is the best way to avoid distractions while driving?", "chosen": "Keep your phone out of reach and focus on the road.", "rejected": "Check your phone while driving to stay updated."},
    {"prompt": "What should you do if your car skids?", "chosen": "Turn the steering wheel in the direction of the skid and gently apply the brakes.", "rejected": "Turn the steering wheel away from the skid and brake hard."},
    {"prompt": "How should you approach a blind curve?", "chosen": "Slow down, stay in your lane, and be prepared to stop.", "rejected": "Drive fast to clear the curve quickly."},
    {"prompt": "How do you maintain a safe driving distance?", "chosen": "Use the 3-second rule to keep a safe distance from the car ahead.", "rejected": "Stay as close as possible to the car ahead to prevent others from cutting in."},
    {"prompt": "What should you do if your windshield wipers aren’t working?", "chosen": "Pull over to a safe location and get the wipers repaired as soon as possible.", "rejected": "Continue driving without using the wipers."},
    {"prompt": "What is the first thing you should do when stopped by a police officer?", "chosen": "Pull over safely, keep your hands visible, and wait for instructions.", "rejected": "Get out of the car and approach the officer."},
    {"prompt": "What should you do if you’re about to be rear-ended?", "chosen": "Try to move forward if possible, and keep your seatbelt on.", "rejected": "Jump out of the car to avoid injury."},
    {"prompt": "What should you do when driving in a heavy fog?", "chosen": "Use low beam headlights, drive slowly, and keep a safe distance.", "rejected": "Use high beam headlights to see better."},
    {"prompt": "How can you ensure your tires are in good condition?", "chosen": "Check tire pressure regularly and ensure tread depth is adequate.", "rejected": "Only check tires when you notice a problem."},
    {"prompt": "What should you do if you see a green arrow at an intersection?", "chosen": "Proceed with caution, as the green arrow indicates you have the right of way.", "rejected": "Stop and wait until the light changes."},
    {"prompt": "What should you do when driving on a slippery road?", "chosen": "Drive slowly, avoid sudden movements, and increase your following distance.", "rejected": "Drive at normal speed to maintain control."},
    {"prompt": "How can you prevent hydroplaning?", "chosen": "Slow down during rain and avoid driving through large puddles.", "rejected": "Drive faster to clear the water quickly."},
    {"prompt": "How should you drive when approaching a school bus with its stop sign out?", "chosen": "Stop and wait until the bus lights turn off and the stop sign retracts.", "rejected": "Continue driving at normal speed to pass the bus."},
    {"prompt": "What should you do when approaching a sharp curve?", "chosen": "Slow down, stay in your lane, and proceed with caution.", "rejected": "Speed up to get through the curve quickly."},
    {"prompt": "What should you do when driving at night?", "chosen": "Use headlights, reduce speed, and be alert for hazards.", "rejected": "Drive faster to reach your destination quicker."},
    {"prompt": "How should you handle driving when you have a limited view of the road?", "chosen": "Slow down, look ahead carefully, and proceed with caution.", "rejected": "Speed up to get through the area quickly."},
    {"prompt": "How do you handle a situation when the road is icy?", "chosen": "Slow down, increase your following distance, and avoid sudden movements.", "rejected": "Drive at normal speed and don’t worry about traction."},
    {"prompt": "How should you behave when driving in a crowded parking lot?", "chosen": "Drive slowly, be aware of pedestrians, and wait for parking spaces to clear.", "rejected": "Drive fast to get a parking spot quickly."},
    {"prompt": "What should you do when passing a cyclist?", "chosen": "Give them enough space and pass slowly.", "rejected": "Speed up and pass as closely as possible."},
    {"prompt": "How do you stay alert while driving for long periods?", "chosen": "Take regular breaks, stay hydrated, and avoid distractions.", "rejected": "Drink caffeinated beverages and keep driving."},
    {"prompt": "What should you do if your car starts skidding?", "chosen": "Steer in the direction you want to go and ease off the accelerator.", "rejected": "Turn the steering wheel in the opposite direction of the skid."},
    {"prompt": "When should you replace your windshield wipers?", "chosen": "When they streak the windshield or make a squeaking noise.", "rejected": "Only when they stop working entirely."},
    {"prompt": "What should you do if you're driving and your engine temperature gauge is rising?", "chosen": "Pull over, turn off the engine, and let it cool down.", "rejected": "Keep driving until the engine stops."},
    {"prompt": "What should you do if you approach a stop sign?", "chosen": "Come to a complete stop and proceed when it's safe.", "rejected": "Slow down and go if no one is around."},
    {"prompt": "What is the proper way to drive on a highway?", "chosen": "Maintain a safe following distance, use the left lane for passing, and signal lane changes.", "rejected": "Always stay in the left lane unless you're passing."},
    {"prompt": "What should you do if your headlights fail while driving?", "chosen": "Use your emergency flashers, pull over, and call for assistance.", "rejected": "Keep driving and use high beam headlights."},
    {"prompt": "How should you prepare your car for winter?", "chosen": "Check the battery, tires, windshield wipers, and antifreeze levels.", "rejected": "Ignore winter preparations if your car is running fine."},
    {"prompt": "What should you do when driving on a wet road?", "chosen": "Reduce your speed and keep a safe distance from other vehicles.", "rejected": "Speed up to avoid water accumulation."},
    {"prompt": "When is it safe to pass another vehicle?", "chosen": "When there's a broken line in the lane and it's clear of oncoming traffic.", "rejected": "Whenever you feel you can make it in time."},
    {"prompt": "How do you handle driving in a construction zone?", "chosen": "Slow down, follow posted speed limits, and be alert for workers and equipment.", "rejected": "Speed up to pass through quickly."},
    {"prompt": "What should you do if you see a pedestrian crossing the road?", "chosen": "Slow down and stop to allow them to cross.", "rejected": "Keep driving without stopping if no cars are behind you."},
    {"prompt": "What should you do if your car overheats?", "chosen": "Pull over safely, turn off the engine, and allow it to cool down before checking the radiator.", "rejected": "Keep driving and try to make it to your destination."},
    {"prompt": "What should you do when you see a flashing yellow light?", "chosen": "Slow down and proceed with caution.", "rejected": "Speed up to pass through quickly."},
    {"prompt": "How can you prevent a collision with a rear-ended vehicle?", "chosen": "Maintain a safe following distance and avoid sudden stops.", "rejected": "Drive close to other cars to avoid being rear-ended."},
    {"prompt": "How should you react if another driver is tailgating you?", "chosen": "Move to another lane or reduce speed slightly to allow them to pass.", "rejected": "Drive faster to avoid them."},
    {"prompt": "What should you do when approaching a roundabout?", "chosen": "Yield to traffic already in the roundabout and proceed when clear.", "rejected": "Drive straight through without yielding."},
    {"prompt": "What should you do when your tire blows out on the highway?", "chosen": "Grip the wheel tightly, slow down gradually, and pull over to a safe spot.", "rejected": "Brake hard and steer into the shoulder immediately."},
    {"prompt": "When should you use your high beam headlights?", "chosen": "Use them when driving in rural areas or when there is no oncoming traffic.", "rejected": "Always use high beams even in well-lit areas."},
    {"prompt": "How should you drive when you are fatigued?", "chosen": "Take frequent breaks, get rest, and avoid driving long hours without rest.", "rejected": "Drink caffeine and keep driving until you reach your destination."},
    {"prompt": "What should you do if you encounter an aggressive driver?", "chosen": "Stay calm, avoid eye contact, and allow them to pass.", "rejected": "Challenge them and drive aggressively back."},
    {"prompt": "What should you do if you’re involved in a minor fender bender?", "chosen": "Exchange information, take photos, and file a report with authorities if necessary.", "rejected": "Drive away quickly before anyone notices."},
    {"prompt": "How should you maintain control when driving in strong winds?", "chosen": "Hold the steering wheel firmly and reduce your speed.", "rejected": "Speed up to get through the windy area faster."},
    {"prompt": "What should you do when driving in foggy conditions?", "chosen": "Use low beam headlights, reduce speed, and maintain a safe distance.", "rejected": "Use high beam headlights to improve visibility."},
    {"prompt": "How do you properly drive through a crosswalk?", "chosen": "Yield to pedestrians and stop if necessary.", "rejected": "Ignore the crosswalk and pass through quickly."},
    {"prompt": "How should you drive when making a U-turn?", "chosen": "Signal in advance, check for traffic, and make the U-turn when safe.", "rejected": "Make a U-turn quickly without checking for oncoming traffic."},
    {"prompt": "What should you do if you have a flat tire?", "chosen": "Pull over to a safe location, use the jack to lift the car, and change the tire.", "rejected": "Keep driving until you reach the nearest service station."},
    {"prompt": "How should you drive when passing a large truck?", "chosen": "Pass quickly and stay visible in the truck's mirrors.", "rejected": "Drive slowly and stay behind the truck."},
    {"prompt": "What is the safest way to handle a left-hand turn?", "chosen": "Signal well in advance, check for traffic, and turn when it's safe.", "rejected": "Turn quickly without checking for oncoming traffic."},
    {"prompt": "What should you do if you feel your car is overheating?", "chosen": "Pull over safely, turn off the engine, and allow the engine to cool down.", "rejected": "Keep driving until the engine cools on its own."},
    {"prompt": "What should you do when approaching an intersection with a green light?", "chosen": "Proceed if the intersection is clear, but be ready to stop.", "rejected": "Drive through quickly without checking for pedestrians or other vehicles."},
    {"prompt": "What is the proper procedure when merging into traffic?", "chosen": "Adjust your speed to match the flow of traffic and merge when it's safe.", "rejected": "Merge immediately regardless of the traffic conditions."},
    {"prompt": "How should you drive in an area with heavy construction?", "chosen": "Slow down, be alert for construction workers, and follow posted signs.", "rejected": "Drive at normal speed and ignore any construction signs."},
    {"prompt": "What should you do when you see an animal crossing the road?", "chosen": "Slow down and be prepared to stop if necessary.", "rejected": "Speed up to avoid hitting the animal."},
    {"prompt": "How should you handle driving on a gravel road?", "chosen": "Slow down, avoid sudden steering movements, and maintain control of your vehicle.", "rejected": "Speed up to get through the gravel quickly."},
    {"prompt": "What should you do when driving through a construction zone?", "chosen": "Slow down, obey the posted signs, and be alert for workers.", "rejected": "Drive at the same speed as usual to avoid delays."},
    {"prompt": "What should you do when driving at night in unfamiliar areas?", "chosen": "Use headlights, reduce speed, and stay alert for potential hazards.", "rejected": "Drive at normal speed, ignoring potential hazards."},
    {"prompt": "What should you do when you see a vehicle stopped on the side of the road?", "chosen": "Slow down, change lanes if possible, and be cautious.", "rejected": "Speed up to pass the vehicle as quickly as possible."},
    {"prompt": "How should you approach a red light?", "chosen": "Slow down and prepare to stop, checking for any pedestrians or vehicles.", "rejected": "Speed up to try and beat the red light."},
    {"prompt": "What should you do when driving through a narrow bridge?", "chosen": "Slow down, check for oncoming traffic, and yield if necessary.", "rejected": "Speed up to pass through the bridge quickly."},
    {"prompt": "What should you do if you are driving and you feel sleepy?", "chosen": "Pull over to a safe place and rest or switch drivers if possible.", "rejected": "Keep driving; you will wake up soon."},
    {"prompt": "How should you handle driving in a hailstorm?", "chosen": "Pull over to a safe location and stay in the car until the storm passes.", "rejected": "Keep driving to get out of the hailstorm faster."},
    {"prompt": "What should you do when driving on an icy road?", "chosen": "Slow down, avoid sudden movements, and keep a safe distance from other vehicles.", "rejected": "Speed up to clear the ice quickly."},
    {"prompt": "What should you do when you see a flashing red traffic light?", "chosen": "Treat it like a stop sign and proceed when safe.", "rejected": "Keep going without stopping."},
    {"prompt": "What should you do if you see a vehicle making an illegal U-turn?", "chosen": "Slow down and avoid the vehicle, staying alert for other drivers.", "rejected": "Try to beat the vehicle and continue driving."},
    {"prompt": "What should you do if you're driving on a road with loose gravel?", "chosen": "Reduce speed and steer gently to maintain control.", "rejected": "Speed up to get through the gravel quickly."}
]

# Convert to column-wise dictionary format
data_dict = {
    "prompt": [d["prompt"] for d in data],
    "chosen": [d["chosen"] for d in data],
    "rejected": [d["rejected"] for d in data],
}

# Create dataset
dataset = Dataset.from_dict(data_dict)
split_data = dataset.train_test_split(test_size=0.2,seed=42)
train_dataset = split_data['train']
test_dataset = split_data['test']

# 2. Load Llama 2 7B with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    max_seq_length=2048,  # Reduce if OOM
    dtype=torch.float16,
    load_in_4bit=True   
)
# dataset = dataset.train_test_split(test_size=0.2)


# 3. Add LoRA adapters (Unsloth-optimized)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
)

# 4. Tokenize dataset
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


# 5. Configure DPO training
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Can increase with Unsloth
    gradient_accumulation_steps=4,
    num_train_epochs = 3,
    learning_rate=5e-5,
    logging_steps=10,
    output_dir="./dpo_results",
    optim="adamw_8bit",  # Unsloth's optimized optimizer
    seed = 42,
    fp16=True,
    remove_unused_columns=False,
      # DPO temperature
)

# 6. Initialize DPOTrainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=split_data['test'],  # use same if no split
    tokenizer=tokenizer,
)
# 7. Train (2-5x faster than vanilla)
dpo_trainer.train()
model.save_pretrained("dpo_finetuned_model")
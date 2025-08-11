import os
import pandas as pd
from datasets import Dataset

from unsloth import FastLanguageModel, is_bfloat16_supported, unsloth_train
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

from config import *


import warnings
warnings.filterwarnings('ignore')   

def main():

    load_in_4bit = False
    load_in_8bit = False
    full_finetuning = False
    if LOAD_BIT == "8":
        load_in_8bit = True
    if LOAD_BIT == "4":
        load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = load_in_4bit,
        load_in_8bit = load_in_8bit,
        full_finetuning = full_finetuning,
        device_map = "balanced",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        target_modules = LORA_TARGET_MODULES,
        bias = LORA_BIAS,
        use_gradient_checkpointing = GRADIENT_CHECKPOINTING,
        random_state = RANDOM_SEED,
        use_rslora = USE_RSLORA,
        loftq_config = LOFTQ_CONFIG
    )

    # Unsloth recommends using their template:
    # https://docs.unsloth.ai/basics/datasets-guide
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "phi-4", # change this to the right chat_template name
    )

    df = pd.read_parquet(DATA_PATH)

    if DEBUG:
        print("DEBUG MODE ON")
        df = df[df['fold'].isin([0,9])]
        df = df.sample(frac=0.1, random_state=RANDOM_SEED)
        df = df.reset_index(drop=True)


    # --- Put prompt in ChatML role/content format ---
    conversations = []
    for r in df.itertuples():

        conversations.append([
            {"role": "user", "content": r.user},
            {"role": "assistant", "content": r.assistant}
        ])

    df['conversations'] = conversations


    # --- Setup train/validation datasets
    trn_df = df[df['fold'] != VALIDATION_FOLD]
    val_df = df[df['fold'] == VALIDATION_FOLD]

    trn_ds = Dataset.from_pandas(trn_df)
    val_ds = Dataset.from_pandas(val_df)


    # --- Create `text` column with appropriate template ---
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize = False, add_generation_prompt = False
            )
            for convo in convos
        ]
        return { "text" : texts, }

    trn_ds = trn_ds.map(
        formatting_prompts_func,
        batched=True,
    )

    val_ds = val_ds.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = trn_ds,
        eval_dataset = val_ds, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            gradient_checkpointing = USE_GRADIENT_CHECKPOINTING,
            per_device_train_batch_size = TRAIN_BATCH_SIZE,
            per_device_eval_batch_size = EVAL_BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS, # Use GA to mimic batch size!
            eval_accumulation_steps = EVAL_ACCUMULATION_STEPS,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            warmup_steps = WARMUP_STEPS,
            eval_strategy = EVAL_STRATEGY,
            eval_steps = EVAL_STEPS,
            save_steps = SAVE_STEPS,
            num_train_epochs = EPOCHS, # Set this for 1 full training run.
            learning_rate = LEARNING_RATE, # Reduce to 2e-5 for long training runs
            logging_steps = LOGGING_STEPS,
            optim = OPTIMIZER,
            weight_decay = WEIGHT_DECAY,
            lr_scheduler_type = LR_SCHEDULER_TYPE,
            seed = RANDOM_SEED,
            load_best_model_at_end = LOAD_BEST_MODEL_AT_END,
            report_to = "wandb",
            output_dir = OUTPUT_DIR,
        ),
    )

    # --- Modifications to Trainer
    
    # Have loss calculated only on the assistant response portion
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user<|im_sep|>",
        response_part="<|im_start|>assistant<|im_sep|>",
    )

    # Use to avoid issues with Gradient Accumulation
    # https://unsloth.ai/blog/gradient
    trainer_stats = unsloth_train(trainer)

    print(trainer_stats)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":

    main()

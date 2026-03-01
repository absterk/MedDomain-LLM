import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from task2_baseline import load_json, build_prompt
from config.model_config import (
    BASE_MODEL_NAME,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_LOW_CPU_MEM_USAGE,
)
from config.path_config import (
    FINETUNED_ADAPTER_DIR,
    FINETUNED_MODEL_DIR,
    TASK2_FINETUNE_CSV,
    TEST_JSON,
    TRAIN_JSON,
    ensure_runtime_dirs,
)
from config.training_config import (
    BATCH_SIZE,
    DO_SAMPLE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MAX_LEN,
    MAX_NEW_TOKENS,
    NUM_EPOCHS,
    REPETITION_PENALTY,
)


BASE_MODEL = BASE_MODEL_NAME

ADAPTER_SAVE_DIR = FINETUNED_ADAPTER_DIR
MERGED_MODEL_SAVE_DIR = FINETUNED_MODEL_DIR      # full model after merging
OUTPUT_CSV = TASK2_FINETUNE_CSV

class DiagnosisDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.items = []
        for row in rows:
            prompt = build_prompt(row["prompt"])
            full_text = prompt + str(row["response"]) + tokenizer.eos_token
            tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LEN, padding="max_length")
            prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LEN)["input_ids"]

            input_ids = torch.tensor(tokenized["input_ids"])
            attention_mask = torch.tensor(tokenized["attention_mask"])
            labels = input_ids.clone()

            # Mask prompt and padding tokens so loss only applies to response
            labels[:len(prompt_ids)] = -100
            labels[attention_mask == 0] = -100

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def generate_text(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            remove_invalid_values=True,
        )

    generated = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return generated


def main():
    ensure_runtime_dirs()

    # Load data
    train_rows = load_json(TRAIN_JSON)
    test_rows = load_json(TEST_JSON)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type=="cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
    )
    model.to(device)
    model.config.use_cache = False

    # Apply LoRA adapter
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS
    )
    model = get_peft_model(model, lora)

 
    # Fine-tune LoRA
    dataset = DiagnosisDataset(train_rows, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_steps = len(loader)
        running_loss = 0.0
        valid_steps = 0
        skipped_steps = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or loss.item() == 0.0:
                skipped_steps += 1
                continue  # skip bad batch

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            loss_value = float(loss.item())
            running_loss += loss_value
            valid_steps += 1

        if valid_steps > 0:
            print(
                f"[train] epoch={epoch + 1}/{NUM_EPOCHS} done "
                f"avg_loss={running_loss / valid_steps:.4f} valid_steps={valid_steps} skipped={skipped_steps}"
            )
        else:
            print(f"[train] epoch={epoch + 1}/{NUM_EPOCHS} done with no valid steps (skipped={skipped_steps})")

    # Save LoRA adapter
    model.save_pretrained(ADAPTER_SAVE_DIR)
    tokenizer.save_pretrained(ADAPTER_SAVE_DIR)
    print(f"LoRA adapter saved: {ADAPTER_SAVE_DIR}")

    # Merge LoRA into full model
    model = model.merge_and_unload()
    model.to(device)
    model.save_pretrained(MERGED_MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_SAVE_DIR)
    print(f"Merged model saved: {MERGED_MODEL_SAVE_DIR}")


    # Generate predictions
    model.eval()
    results = []
    for i, row in enumerate(test_rows, start=1):
        prompt = build_prompt(row["prompt"])
        gen = generate_text(model, tokenizer, prompt)
        print(f"[{i}/{len(test_rows)}] Generated: {gen}")
        results.append({"prompt": row["prompt"], "reference": row["response"], "generated": gen})

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

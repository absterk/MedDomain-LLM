import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.model_config import BASE_MODEL_NAME, MODEL_DEVICE_MAP, MODEL_LOW_CPU_MEM_USAGE
from config.path_config import BASELINE_MODEL_DIR, TASK2_BASELINE_CSV, TEST_JSON, ensure_runtime_dirs
from config.training_config import DO_SAMPLE, MAX_LEN, MAX_NEW_TOKENS, REPETITION_PENALTY

# ----------------------
# Configuration
# ----------------------
BASE_MODEL = BASE_MODEL_NAME
OUTPUT_CSV = TASK2_BASELINE_CSV
BASELINE_SAVE_DIR = BASELINE_MODEL_DIR

# ----------------------
# Helper Functions
# ----------------------
def load_json(path):
    """Load a JSON file and return its contents as a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(findings: str) -> str:
    """Format a findings string into a model prompt."""
    return f"Findings:\n{findings}\n\nImpression:\n"


def load_model_and_tokenizer(model_ref):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        device_map=MODEL_DEVICE_MAP,
        torch_dtype=dtype,
        low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE,
    )
    model.eval()
    return model, tokenizer


def save_model_snapshot(model, tokenizer, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def generate_predictions(model, tokenizer, test_rows):
    device = next(model.parameters()).device
    rows = []

    for row in test_rows:
        prompt = build_prompt(row["prompt"])
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        rows.append(
            {
                "prompt": row["prompt"],
                "reference": row["response"],
                "generated": generated,
            }
        )

    return rows


# ----------------------
# Main function
# ----------------------
def main():
    # Ensure output directories exist
    ensure_runtime_dirs()

    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL)

    # Save baseline model for reproducibility
    save_model_snapshot(model, tokenizer, BASELINE_SAVE_DIR)
    print(f"Baseline model saved at: {BASELINE_SAVE_DIR}")

    # Load test dataset
    test_data = load_json(TEST_JSON)

    # Generate impressions
    results = generate_predictions(model=model, tokenizer=tokenizer, test_rows=test_data)

    # Save all predictions to CSV
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Baseline predictions saved to: {OUTPUT_CSV}")


# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task2_baseline import build_prompt, load_json
from config.path_config import FINETUNED_MODEL_DIR, TASK2_GENERATED_RESULTS_CSV, TEST_JSON, ensure_runtime_dirs
from config.training_config import DO_SAMPLE, MAX_LEN, MAX_NEW_TOKENS, REPETITION_PENALTY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with merged fine-tuned model")
    parser.add_argument("--input-data", default=str(TEST_JSON), help="Path to JSON test data")
    parser.add_argument("--model-path", default=str(FINETUNED_MODEL_DIR), help="Path to fine-tuned model")
    parser.add_argument("--output-path", default=str(TASK2_GENERATED_RESULTS_CSV), help="Path to output CSV")
    return parser.parse_args()


def generate_predictions(model, tokenizer, test_rows):
    device = next(model.parameters()).device
    rows = []

    for row in test_rows:
        prompt = build_prompt(row["prompt"])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN)
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
                remove_invalid_values=True,
            )

        generated = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        rows.append({"prompt": row["prompt"], "reference": row["response"], "generated": generated})

    return rows


def main() -> None:
    args = parse_args()
    ensure_runtime_dirs()

    model_path = str(args.model_path)
    input_data_path = str(args.input_data)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_rows = load_json(input_data_path)
    results = generate_predictions(model, tokenizer, test_rows)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"Inference complete. Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()

# task 1.2 LLM as LLM-as-a-judge to design a simple multi-agent system to compare model ouput and reference standard and give quantiative scores
# Metrics to score:
# 1. Scientific terminology: Are terms precise and medically correct?
# 2. Coherence: Is the impression logically structured and understandable?
# 3. Specific diagnosis: Does it mention exact disease/condition/radiologic sign?
# 4. Differential diagnosis: Are relevant alternative diagnoses mentioned (if appropriate)?
# 5. Management recommendations: Are follow-ups or exams suggested appropriately?
# 6. Correctness: Are the impressions supported by the imaging findings?
# 7. Comprehensiveness: Does the impression cover all relevant findings?
# 8. Harmlessness / Lack of bias: No statements that are harmful or could bias interpretation
# Use the same model as baseline (m42-health/Llama3-Med42-8B) to act as a judge
# Prompt the model to compare generated vs reference impressions
# Ask the model to return ONLY a Python dictionary with the scores
# Make function evaluate_impression(generated, reference) that returns the scores as a dictionary
# Input: task1.1_generated_impressions.csv with case_id, generated_impression, output_impression
# Output: evaluation_metrics.csv with case_id, generated_impression, output_impression, and one column per score metric

import pandas as pd
import torch
import ast
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from config.model_config import (
    BASE_MODEL_NAME,
    MODEL_DEVICE_MAP,
    MODEL_LOW_CPU_MEM_USAGE,
    TOKENIZER_PADDING_SIDE,
)
from config.path_config import TASK1_GENERATED_IMPRESSIONS_CSV, TASK1_SCORES_CSV, ensure_runtime_dirs
from config.training_config import DO_SAMPLE, GENERATION_BATCH_SIZE, MAX_JUDGE_NEW_TOKENS

# 1) Load generated results with ground-truth impressions
df = pd.read_csv(TASK1_GENERATED_IMPRESSIONS_CSV)


# 2) Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 3) Load the Med42 LLaMA3 model (Hugging Face)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, padding_side=TOKENIZER_PADDING_SIDE)
dtype = torch.float16 if device == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map=MODEL_DEVICE_MAP,           # Hugging Face accelerate GPU mapping
    torch_dtype=dtype,
    low_cpu_mem_usage=MODEL_LOW_CPU_MEM_USAGE
) 

# 4) Create Hugging Face pipeline
evaluator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 5) Function to evaluate impressions
def build_eval_prompt(generated, reference):
    return f"""
    You are a clinical radiology AI evaluator.

Compare the following AI-generated impressions with the reference impressions:

Generated: {generated}
Reference: {reference}

Score the generated impressions on a scale from 0 (worst) to 5 (best) for each metric:

1. Scientific terminology – Are all terms medically accurate and precise?
2. Coherence – Is the impression logically structured and easy to understand?
3. Specific diagnosis – Does it identify a clear specific disease or radiologic sign?
4. Differential diagnosis – Are relevant alternative diagnoses mentioned (if appropriate)?
5. Management recommendations – Are follow-up tests or exams suggested appropriately?
6. Correctness – Are the impressions supported by the imaging findings?
7. Comprehensiveness – Does the impression cover all relevant findings?
8. Harmlessness / Lack of bias – Is the impression safe and free of bias?

Return your scores **only as a Python dictionary**, using integers 0–5 for each metric.  
Do not add explanations or extra text.

Example output:
{{"scientific_terminology": 5, "coherence": 4, "specific_diagnosis": 5, "differential_diagnosis": 4, "management_recommendations": 3, "correctness": 5, "comprehensiveness": 5, "harmlessness_bias": 5}}
"""


SCORE_COLUMNS = [
    "scientific_terminology",
    "coherence",
    "specific_diagnosis",
    "differential_diagnosis",
    "management_recommendations",
    "correctness",
    "comprehensiveness",
    "harmlessness_bias",
]


def parse_scores(output_text):
    try:
        match = re.search(r"\{[\s\S]*\}", output_text.strip())
        if not match:
            raise ValueError("No dictionary found in model output.")
        raw_scores = ast.literal_eval(match.group(0))
        if not isinstance(raw_scores, dict):
            raise ValueError("Parsed score output is not a dictionary.")
    except Exception:
        raw_scores = {}

    normalized_scores = {}
    for key in SCORE_COLUMNS:
        value = raw_scores.get(key)
        try:
            value = int(value)
            value = max(0, min(5, value))
        except (TypeError, ValueError):
            value = None
        normalized_scores[key] = value
    return normalized_scores

# 6) Evaluate each impression in batches using a dataset
prompt_df = pd.DataFrame({
    "prompt": df.apply(
    lambda row: build_eval_prompt(row["generated_impression"], row["output_impression"]),
    axis=1
)})
prompt_dataset = Dataset.from_pandas(prompt_df, preserve_index=False)

generated_outputs = evaluator(
    KeyDataset(prompt_dataset, "prompt"),
    max_new_tokens=MAX_JUDGE_NEW_TOKENS,
    return_full_text=False,
    do_sample=DO_SAMPLE,
    batch_size=GENERATION_BATCH_SIZE,
)

scores_series = pd.Series(
    [parse_scores(item[0]["generated_text"]) for item in generated_outputs],
    index=df.index
)
scores_df = pd.json_normalize(scores_series)
df = pd.concat([df, scores_df], axis=1)

results_df = df[["case_id", "generated_impression", "output_impression", *SCORE_COLUMNS]].copy()

# 7) Save results to csv
ensure_runtime_dirs()
results_df.to_csv(TASK1_SCORES_CSV, index=False)
print(f"Saved {len(results_df)} rows to {TASK1_SCORES_CSV}")

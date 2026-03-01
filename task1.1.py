# Python script to:
# 1. Load xlsx with radiology findings
# 2. Send each input_findings to an open-source medical LLM (Med42 LLaMA3 model)
# 3. Generate structured impressions
# 4. Save outputs to task1.1_generated_impressions.csv with case_id, and generated_impression columns
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
from config.path_config import (
    TASK1_GENERATED_IMPRESSIONS_CSV,
    TRAIN_TEST_XLSX,
    ensure_runtime_dirs,
)
from config.training_config import DO_SAMPLE, GENERATION_BATCH_SIZE, MAX_JUDGE_NEW_TOKENS


# 1) Load test sheet from xlsx file
df = pd.read_excel(TRAIN_TEST_XLSX, sheet_name="test")

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
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# 5) Function to generate structured impressions
def build_prompt(findings):
    return f"""
You are a clinical radiology AI assistant.
Task: generate disease impressions based on free-text radiology findings.   
Rules:
- Output only as Python list of strings.
- Each string should be a concise disease impression.
- Do not include any explanations or additional text.
- Use formal medical terminology, consistent with the examples provided.
Examples:
['Multiple renal cysts.']
['Ascending colon cancer.', 'Multiple liver metastases.', 'Small cysts in the right lobe of the liver.']
['Mild hepatic steatosis.', 'Small stones or calcifications in the right kidney, slightly perirenal effusion bilaterally.']
Input Findings:
{findings}
Output:
"""


def parse_impression(output):
    # Keep only one list output and normalize it to a clean list[str].
    text = output.split("Output:")[-1].strip()
    matches = re.findall(r"\[[\s\S]*?\]", text)
    if not matches:
        return []

    for candidate in matches:
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            continue
    return []


# 6) Generate impressions in batches
prompt_df = pd.DataFrame({"prompt": df["input_findings"].apply(build_prompt)})
prompt_dataset = Dataset.from_pandas(prompt_df, preserve_index=False)

generated_outputs = generator(
    KeyDataset(prompt_dataset, "prompt"),
    max_new_tokens=MAX_JUDGE_NEW_TOKENS,
    return_full_text=False,
    do_sample=DO_SAMPLE,
    batch_size=GENERATION_BATCH_SIZE,
)

df["generated_impression"] = [
    parse_impression(item[0]["generated_text"]) for item in generated_outputs
]

# 7) Save full test data + generated impressions to CSV
ensure_runtime_dirs()
df.to_csv(TASK1_GENERATED_IMPRESSIONS_CSV, index=False)
print(f"Results saved to {TASK1_GENERATED_IMPRESSIONS_CSV}")

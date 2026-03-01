"""Model and adapter configuration constants."""

BASE_MODEL_NAME = "m42-health/Llama3-Med42-8B"
MODEL_DEVICE_MAP = "auto"
MODEL_LOW_CPU_MEM_USAGE = True
TOKENIZER_PADDING_SIDE = "left"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_BIAS = "none"

"""Project paths and output filenames."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_GENERATED_DIR = OUTPUT_DIR / "generated"
OUTPUT_METRICS_DIR = OUTPUT_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

BASELINE_EXPERIMENT_DIR = EXPERIMENTS_DIR / "baseline"
FINETUNE_EXPERIMENT_DIR = EXPERIMENTS_DIR / "finetune"
COMPARISON_EXPERIMENT_DIR = EXPERIMENTS_DIR / "comparison"

TRAIN_TEST_XLSX = DATA_RAW_DIR / "train-test-data.xlsx"
TRAIN_JSON = DATA_RAW_DIR / "train.json"
TEST_JSON = DATA_RAW_DIR / "test.json"

TASK1_GENERATED_IMPRESSIONS_CSV = OUTPUT_GENERATED_DIR / "task1.1_generated_impressions.csv"
TASK1_SCORES_CSV = OUTPUT_METRICS_DIR / "task1.2_scores.csv"
TASK2_GENERATED_RESULTS_CSV = OUTPUT_GENERATED_DIR / "task2_generated_results.csv"

TASK2_BASELINE_CSV = BASELINE_EXPERIMENT_DIR / "task2_baseline.csv"
TASK2_FINETUNE_CSV = FINETUNE_EXPERIMENT_DIR / "task2_finetune.csv"
COMPARISON_SUMMARY_CSV = COMPARISON_EXPERIMENT_DIR / "comparison_metrics.csv"
COMPARISON_DETAIL_CSV = COMPARISON_EXPERIMENT_DIR / "comparison_metrics_detailed.csv"

BASELINE_MODEL_DIR = MODELS_DIR / "baseline_model"
FINETUNED_MODEL_DIR = MODELS_DIR / "finetuned_model"
FINETUNED_ADAPTER_DIR = MODELS_DIR / "finetuned_model_adapter"


def ensure_runtime_dirs() -> None:
    """Create runtime directories used by scripts."""
    for directory in (
        OUTPUT_DIR,
        OUTPUT_GENERATED_DIR,
        OUTPUT_METRICS_DIR,
        MODELS_DIR,
        BASELINE_EXPERIMENT_DIR,
        FINETUNE_EXPERIMENT_DIR,
        COMPARISON_EXPERIMENT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

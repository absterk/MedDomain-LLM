# Medical Radiology Impressions Generation using Llama3-Med42-8B + LoRA

## Project Description

This project builds a clinical text-generation system that converts free-text radiology findings into structured radiology impressions using `m42-health/Llama3-Med42-8B`, fine-tuned with LoRA (Low-Rank Adaptation) for parameter-efficient adaptation.  
The workflow supports reproducible training, inference, and evaluation while integrating AI-assisted development practices with GitHub Copilot, ChatGPT, and Codex for coding support, debugging, and technical writing acceleration.

## Environments and Requirements

- OS: Windows / Ubuntu
- CPU: 8+ logical cores recommended
- RAM: 16 GB minimum (32 GB recommended)
- GPU: NVIDIA GPU with 16 GB+ VRAM recommended for fine-tuning
- CUDA: 12.x recommended
- Python: 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

You can use `venv`, Conda, or a system Python environment. A standard `venv` flow is shown below:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Conda alternative:

```bash
conda create -n med42-lora python=3.10 -y
conda activate med42-lora
pip install -r requirements.txt
```

## Dataset

The training/evaluation dataset is JSON-based and follows an instruction-response format:

- `prompt`: free-text radiology findings
- `response`: structured radiology impression

No manual preprocessing of clinical text is required before training.

Example record:

```json
{
  "prompt": "Findings: ...",
  "response": "Impression: ..."
}
```

## Preprocessing

No separate preprocessing is required. Tokenization and masking are handled in the dataset class.

## Training

### Training with LoRA

Run LoRA fine-tuning with:

```bash
python finetune.py
```

LoRA is used as a parameter-efficient fine-tuning method: instead of updating all base-model parameters, low-rank adapter matrices are trained in selected attention layers, reducing memory and compute requirements while preserving strong downstream adaptation quality.

Note: training hyperparameters and paths are configured in `config/training_config.py`, `config/model_config.py`, and `config/path_config.py`.

## Fine-tuning on Custom Dataset

Set your custom dataset path in `config/path_config.py` (for example `TRAIN_JSON`), then run:

```bash
python finetune.py
```

## Inference

Generate predictions from a fine-tuned/merged model:

```bash
python inference.py --input-data data/raw/test.json --model-path models/finetuned_model/ --output-path output/generated/task2_generated_results.csv
```

Inference notes:

- Greedy decoding is used for deterministic generation.
- Repetition penalty is applied to reduce looping/redundant outputs.
- `remove_invalid_values` is enabled to improve generation robustness.
- Predictions are saved to the `--output-path` CSV.

## Evaluation

```bash
python eval.py
```

To evaluate different prediction/reference files, update these variables in `config/path_config.py` before running `eval.py`:

- `TASK2_BASELINE_CSV`: path to baseline predictions CSV
- `TASK2_FINETUNE_CSV`: path to fine-tuned predictions CSV
- `COMPARISON_SUMMARY_CSV`: path for aggregated metric output
- `COMPARISON_DETAIL_CSV`: path for per-sample metric output

Expected input CSV format for both baseline and fine-tuned files:

- `prompt`
- `reference`
- `generated`

Evaluation metrics:

- EM (Exact Match)
- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L

## Results

Baseline vs LoRA fine-tuned results (`n=380`):

| Model | EM | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0000 | 0.0456 | 0.1895 | 0.0900 | 0.1770 |
| Fine-tuned (LoRA) | 0.0263 | 0.1418 | 0.4648 | 0.2586 | 0.4522 |

## AI-assisted Workflow

- GitHub Copilot for boilerplate acceleration and iterative code completion.
- ChatGPT/Codex for debugging support, error analysis, and rapid refactoring guidance.
- AI-assisted drafting for experiment documentation and report writing.

## Known Issues and Fixes

- Degenerate outputs (for example `"!!!!"`):
  - Mitigated by decoding constraints and repetition penalty tuning.
- NaN loss during training:
  - Addressed with safer optimization settings and invalid batch handling.
- Label masking issues:
  - Fixed by correcting prompt/response token masking boundaries.
- Invalid/empty batches:
  - Handled by skipping malformed samples during collation/training.
- Generation robustness issues:
  - Improved with `remove_invalid_values`, stable decoding config, and post-generation sanitation.

## Docker

Example Docker workflow:

```bash
docker build -t med42-lora .
docker run --gpus all --rm -it -v $(pwd):/workspace med42-lora
```

## Colab

This pipeline can also be executed in Google Colab by:

1. Uploading the dataset and scripts.
2. Installing dependencies from `requirements.txt`.
3. Running fine-tuning and inference cells in sequence.
4. Exporting predictions/metrics to Drive or local download.

## Trained Models

Typical model artifacts:

- LoRA adapter: `models/finetuned_model_adapter/`
- Merged fine-tuned model: `models/finetuned_model/`

You may also keep alternative paths such as:

- Base model: `saved_models/base_model/`
- Merged model: `saved_models/finetuned_model_merged/`

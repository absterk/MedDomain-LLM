# Medical Radiology Impressions Generation using Llama3-Med42-8B + LoRA

This repository is the official implementation of [**Automated Radiology Impression Generation Using a Fine-Tuned Medical Language Model**](./medicalllm_report.pdf).


## Environments and Requirements

- OS: Windows / Ubuntu
- CPU: 8+ logical cores
- RAM: 16 GB minimum (32 GB recommended)
- GPU: NVIDIA GPU with 12+ GB VRAM recommended
- CUDA: 11.8+
- Python: 3.10+

To install requirements:

```bash
pip install -r requirements.txt
```

Environment setup examples:

```bash
# venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# conda
conda create -n med42-lora python=3.10 -y
conda activate med42-lora
pip install -r requirements.txt
```

## Dataset

- Location in this repo: `data/raw/`
- Data format: JSON instruction-response pairs with:
  - `prompt`: free-text radiology findings
  - `response`: structured radiology impression

Current folder structure:

```text
data/
  raw/
    train-test-data.xlsx
    train.json
    test.json
```

Example sample:

```json
{
  "prompt": "Findings: ...",
  "response": "Impression: ..."
}
```

## Preprocessing

No clinical-text preprocessing is required for Task 2.

The only preprocessing step is converting/splitting the Excel file into JSON:
- Read `data/raw/train-test-data.xlsx`
- Split into train/test
- Save to `data/raw/train.json` and `data/raw/test.json`

Run:

```bash
python task2_preprocessing.py
```

## Training

To train the model in this repository:

```bash
python finetune.py
```

Training notes:
- Fine-tuning method: LoRA (parameter-efficient adaptation)
- Main configs are in:
  - `config/training_config.py`
  - `config/model_config.py`
  - `config/path_config.py`

## Trained Models

For inference in this repository, download:
- [finetuned_model](https://drive.google.com/drive/folders/1VvoJLkZDfNwDiCNbRdSPNUbD0vqKvESV?usp=drive_link) (merged model)

After downloading, place it in:
- `models/finetuned_model/`

## Inference

To infer testing cases, run:

```bash
python inference.py --input-data data/raw/test.json --model-path models/finetuned_model/ --output-path output/generated/task2_generated_results.csv
```

Inference notes:
- Greedy decoding for deterministic output
- Repetition penalty to reduce redundant text
- `remove_invalid_values` enabled for robustness

## Evaluation

To compute evaluation metrics, run:

```bash
python eval.py
```

To evaluate different files, update paths in `config/path_config.py`:
- `TASK2_BASELINE_CSV`
- `TASK2_FINETUNE_CSV`
- `COMPARISON_SUMMARY_CSV`
- `COMPARISON_DETAIL_CSV`

Expected CSV columns:
- `prompt`
- `reference`
- `generated`

Reported metrics:
- EM (Exact Match)
- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L

## Results

Our method achieves the following performance:

| Model name | EM | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0000 | 0.0456 | 0.1895 | 0.0900 | 0.1770 |
| My model (LoRA fine-tuned) | 0.0263 | 0.1418 | 0.4648 | 0.2586 | 0.4522 |

## Acknowledgement

We acknowledge the creators of the pretrained medical language model **m42-health/Llama3-Med42-8B**, which served as the foundation model for this work.

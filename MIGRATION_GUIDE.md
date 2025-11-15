# Migration Guide: Notebooks to Python Modules

This document describes the conversion of Jupyter notebooks to proper Python modules.

## Changes Made

### 1. Removed Jupyter Notebooks

All `.ipynb` files have been removed from the repository:

- `dataset_creation/all-perturb.ipynb`
- `dataset_creation/clean_dataset.ipynb`
- `dataset_creation/salient_perturb.ipynb`
- `dataset_creation/sent-and-word-perturb.ipynb`
- `experiment and evaluation/claude-classification.ipynb`
- `experiment and evaluation/claude-generation.ipynb`
- `experiment and evaluation/gpt-4o-classification.ipynb`
- `experiment and evaluation/gpt-4o-generation.ipynb`

### 2. Created Python Modules

#### Dataset Creation Module (`dataset_creation/`)

| File                     | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `dataset_utils.py`       | Core utilities for dataset preprocessing, statistics, and cleaning |
| `perturbation.py`        | Text perturbation functions (word, sentence, salient)              |
| `salient_detection.py`   | BanglaBert-based salient word detection                            |
| `generate_indices.py`    | Generate random indices for perturbation                           |
| `add_salient_words.py`   | Add salient words to datasets                                      |
| `apply_perturbations.py` | Main script to apply all perturbations                             |
| `clean_dataset.py`       | Clean and format dataset indices                                   |

#### Experiment & Evaluation Module (`experiment_evaluation/`)

| File                    | Description                             |
| ----------------------- | --------------------------------------- |
| `evaluation_metrics.py` | BLEU, ROUGE, and classification metrics |
| `llm_classification.py` | LLM-based classification (Claude/GPT)   |
| `llm_generation.py`     | LLM-based generation/summarization      |
| `prompts.py`            | Prompt templates for all tasks          |

#### Example Scripts (`examples/`)

| File                           | Description                        |
| ------------------------------ | ---------------------------------- |
| `run_dataset_creation.py`      | Complete dataset creation pipeline |
| `run_claude_classification.py` | Run classification with Claude     |
| `run_gpt_classification.py`    | Run classification with GPT        |
| `run_claude_generation.py`     | Run generation with Claude         |
| `run_gpt_generation.py`        | Run generation with GPT            |

## Usage Comparison

### Before (Notebooks)

```python
!pip install transformers
!pip install anthropic

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

### After (Python Modules)

```python
from dataset_creation import perturb_random_words, BanglaBertAttentionAnalyzer
from experiment_evaluation import run_claude_classification, BANGLA_HATE_SPEECH_PROMPT

analyzer = BanglaBertAttentionAnalyzer(prob=0.2)
salient_words = analyzer.get_salient_words(text)
```

## Command Line Interface

### Generate Random Indices

```bash
python dataset_creation/generate_indices.py \
    --input data/input.csv \
    --output data/output.csv \
    --text_column content \
    --prob 20
```

### Apply All Perturbations

```bash
python dataset_creation/apply_perturbations.py \
    --input data/input.csv \
    --output data/output.csv \
    --text_column content \
    --batch_size 32
```

### Clean Dataset

```bash
python dataset_creation/clean_dataset.py \
    --files data/file1.csv data/file2.csv
```

## Key Improvements

1. **Modularity**: Code is now organized into reusable modules
2. **Maintainability**: Easier to test, debug, and maintain
3. **Version Control**: Better git diffs and collaboration
4. **Command Line**: Scripts can be run from terminal
5. **Imports**: Proper Python package structure with `__init__.py`
6. **Documentation**: Clear README and examples
7. **No Comments**: Clean code without unnecessary comments

## Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Running Examples

```bash
cd examples
python run_dataset_creation.py
python run_claude_classification.py
python run_gpt_generation.py
```

## API Keys

Update API keys in example scripts or use environment variables:

```python
import os
claude_key = os.environ.get('CLAUDE_API_KEY')
gpt_key = os.environ.get('OPENAI_API_KEY')
```

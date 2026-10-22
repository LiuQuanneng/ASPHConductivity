# ASPHConductivity

Data and code for **Topological descriptors for improved prediction of ionic conductivity in solid-state electrolytes**.

Repository: https://github.com/LiuQuanneng/ASPHConductivity

## Results

| CatBoost descriptor set | MAE | R² |
|---|---:|---:|
| An | 1.296 | 0.388 |
| Can | 1.240 | 0.427 |
| An + Can | 1.071 | 0.551 |
| Magpie | 0.853 | 0.681 |
| Magpie + An | 0.810 | 0.714 |
| Magpie + Can | 0.822 | 0.703 |
| Magpie + An + Can | 0.804 | 0.721 |

The full CatBoost descriptor set reduces MAE by 5.7% and increases R² by 5.8% relative to the Magpie-only CatBoost baseline. Percentages use unrounded means. Results for all six algorithms and seven descriptor sets are in [Table S2](publication_fixed_S1/data/Table_S2_all42_fixed_S1.csv).

## Installation

Python 3.12 was used for the supplied results.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-tested-reproduction.txt
```

For structure-based descriptor generation, also install `requirements-descriptors.txt`.

## Run

```bash
# Check scores, candidate predictions and model artifacts
python scripts/check_fixed_S1_publication.py

# Evaluate all six algorithms and seven descriptor sets with fixed Table S1 parameters
python scripts/reproduce_paper_fixed_S1.py --execute --output-dir runs/evaluation

# Predict candidate conductivities
python scripts/predict_fixed_S1.py --features publication_fixed_S1/data/candidate_features_307.csv --output runs/predictions.csv

# Generate plots from the saved results and model
python scripts/regenerate_paper_figures.py
```

Training supports `--combination`, `--models`, `--resume`, `--max-seconds`, and `--fit-final`. Outputs are written to the selected run directory. The default evaluation uses the supplied preprocessing artifacts. `src/original/` contains the underlying training implementations; `scripts/reproduce_paper_fixed_S1.py` is the fixed-parameter evaluation entry point.

## Files

| Directory | Contents |
|---|---|
| `data/features/` | Seven feature matrices with 497 aligned records |
| `data/labels/` | Experimental labels, record identifiers and literature references |
| `src/descriptors/` | Structure processing, persistent homology, topology statistics and Magpie generation |
| `config/` | Table S1 hyperparameters |
| `publication_fixed_S1/preprocessing/` | Scalers, selected feature names and preprocessed arrays |
| `publication_fixed_S1/model/` | Final full-descriptor CatBoost model and matching feature metadata |
| `publication_fixed_S1/data/` | Figure 3 data, 42 model scores, 1,050 fold scores, feature definitions and candidate predictions |
| `publication_fixed_S1/figures/` | Figures and the editable original-style Figure 3 PPTX |

See [methods and data](docs/METHODS.md) for the evaluation procedure and descriptor-generation inputs, and [attribution](docs/ATTRIBUTION.md) for data and method sources. 中文说明见 [README.zh-CN.md](README.zh-CN.md).

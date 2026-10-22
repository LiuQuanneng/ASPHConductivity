# Methods and data

## Training data

The master matrix `data/features/Magpie-AnCan.csv` contains 497 records, 307 descriptors, an ID column and the target `Ionic Conductivity (Log)`. The target is the stored log10 conductivity in S/cm. Source identifiers and literature DOIs are in `data/labels/training_labels_497.csv`.

## Evaluation

The six algorithms use the fixed parameters in `config/manuscript_table_S1_hyperparameters.json`. Scores are means over five repeats of five folds, with split seeds 42–46. Absolute Pearson correlation filtering at 0.95, StandardScaler and supervised XGBoost SHAP top-30% selection are fitted to the full dataset before the fold splits. XGBoost and CatBoost receive the held-out fold as `eval_set`; CatBoost uses its default best-iteration selection. The supplied scores describe this internal evaluation procedure.

For the full descriptor set, correlation filtering retains 195 features and SHAP selection retains 58. Ordered columns and fitted preprocessing are stored under `publication_fixed_S1/preprocessing/`. Final-model SHAP values are calculated using the CatBoost fit to all 497 records.

## Descriptor sets

| Set | Descriptor blocks | Input columns |
|---|---|---:|
| An | Li topology | 35 |
| Can | O/P/S topology | 105 |
| AnCan | Li + O/P/S topology | 140 |
| Magpie | Composition | 132 |
| Magpie-An | Composition + global + Li topology | 202 |
| Magpie-Can | Composition + global + O/P/S topology | 272 |
| Magpie-AnCan | Composition + global + Li + O/P/S topology | 307 |

## Topological descriptor generation

`src/descriptors/feature_engineering.py` reads structures, expands periodic neighborhoods, builds Li–Li and selected anion point clouds, computes persistence barcodes with Ripser up to dimension 2, and extracts global and element-centered statistical descriptors. The neighborhood cutoff is 8.0 Å.

The input directory contains `structure/<ID>.txt` in POSCAR format, `properties.txt`, and `element_properties.csv`. Set `ASPH_DATA_DIR` to this directory before running the generator. The elemental-property resource is supplied in `data/descriptor_resources/`. Crystal structure files are obtained separately from their source databases.

```powershell
$env:ASPH_DATA_DIR = 'D:\research\structure_inputs'
python src/descriptors/feature_engineering.py
```

`barcode_statistics_fixed_S1.py` extracts the 175 global and Li/O/P/S statistics from persistence barcodes. `material_features_from_csv.py` generates composition-based Magpie descriptors from a formula column. Seven model-ready matrices are supplied in `data/features/`.

## Candidate predictions

`candidate_features_307.csv` contains 3,082 MP identifiers and 307 named inputs. Prediction applies the stored scaler and selected column order. For a logarithmic model prediction y, conductivity in mS/cm is `1000 * 10**y`. `candidate_predictions_fixed_S1.csv` includes the predictions and MP metadata; `Table_S4_top50_fixed_S1.csv` contains the fifty highest predictions ranked before rounding.

## Figures and tables

Figure 3 uses `Fig3_CatBoost_fixed_S1.csv`. Table S2 uses `Table_S2_all42_fixed_S1.csv`, with fold-level values in `all42_fold_scores_1050.csv`. Tables S3 and S5 provide feature definitions and descriptor-set counts. `Fig3_original_style.pptx` contains editable Figure 3 objects with the current point coordinates and labels.

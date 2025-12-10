## Elastic Modulus Prediction from Composition & Symmetry

End-to-end Python project for ME 490: AI in Materials Science. The workflow tests the research question **“Can elastic modulus be accurately predicted from compositional and crystal-symmetry descriptors without explicit structural information?”** using the Materials Project elasticity dataset, matminer featurization, and two supervised regression models (Random Forest and MLP).

### Objectives & Scope
- **Dataset**: Elasticity entries (~10k compounds) pulled via the Materials Project API (`MP_API_KEY` required).
- **Target**: Voigt-Reuss-Hill averaged Young’s modulus computed from the elastic tensor.
- **Features**: Matminer elemental fractions & Magpie statistics, crystal system one-hots, binned space-group indicators, and simple heuristics (mean atomic number, average valence electrons).
- **Models**: Scikit-learn RandomForestRegressor and MLPRegressor pipelines, tuned with 5-fold CV on the training split.
- **Evaluation**: MAE, RMSE, and R² on an 80/20 train/test split, plus per-crystal-system breakdowns, prediction/residual plots, and RF feature importance visualization.

### Repository Layout
```
elastic-modulus-ml/
├── data/
│   ├── raw/                 # Materials Project downloads (CSV/JSON)
│   └── processed/           # Cleaned tables, feature matrices, cached splits, figures/
├── models/                  # Serialized estimators + CV results
├── scripts/
│   ├── run_download.py      # Download-only helper
│   ├── run_train.py         # Clean → featurize → train
│   ├── run_evaluate.py      # Evaluate existing models
│   └── run_full_pipeline.py # Download → preprocess → featurize → train → eval
├── src/
│   ├── config.py            # Paths, random seeds, hyperparameter grids
│   ├── data_download.py     # Materials Project query logic
│   ├── preprocessing.py     # Cleaning + VRH modulus computation
│   ├── features.py          # matminer featurization + symmetry encodings
│   ├── models.py            # Model/pipeline constructors
│   ├── train.py             # Cross-validated training + artifact caching
│   ├── evaluate.py          # Metric computation + plots
│   ├── visualization.py     # Matplotlib/seaborn plot helpers
│   └── utils.py             # Logging, seeding, env utilities
├── README.md
└── requirements.txt
```

### Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

**Materials Project API key (mine is provided out of the box)**
- Generate a key at https://materialsproject.org/dashboard.
- Export it as an environment variable (or place it in a `.env` file; `python-dotenv` is loaded automatically):
  ```powershell
  setx MP_API_KEY "your_key_here"     # Windows PowerShell
  ```
  ```bash
  export MP_API_KEY=your_key_here     # macOS / Linux
  ```

### Typical Workflow
1. **Download raw elasticity data**
   ```bash
   python scripts/run_download.py
   # or python -m src.data_download --output data/raw/elasticity_raw.csv
   ```
2. **Clean + featurize + train**
   ```bash
   python scripts/run_train.py
   # runs preprocessing → feature engineering → model training
   ```
3. **Evaluate trained models**
   ```bash
   python scripts/run_evaluate.py
   # computes metrics, per-crystal summaries, and plots into data/processed/figures/
   ```
4. **One-shot end-to-end pipeline**
   ```bash
   python scripts/run_full_pipeline.py
   # orchestrates download → clean → features → train → evaluate (skips cached steps)
   ```
5. **Clean up generated artifacts**
   ```bash
   python scripts/run_cleanup.py --dry-run        # preview deletions
   python scripts/run_cleanup.py --recreate-dirs  # delete + recreate empty data/ + models/
   ```
   Removes everything under `data/raw/`, `data/processed/`, and `models/` so you can free space or restart from scratch.

Key artifacts:
- `data/raw/elasticity_raw.csv`: raw Materials Project pull.
- `data/processed/elasticity_cleaned.csv`: cleaned table with VRH modulus.
- `data/processed/elasticity_features.parquet`: feature matrix + target.
- `data/processed/test_set.joblib`: cached deterministic 20% test split.
- `models/random_forest.joblib`, `models/mlp.joblib`: tuned estimators.
- `data/processed/figures/`: prediction vs. truth, residuals, target distribution, RF feature importances, etc.

**Note**: Due to GitHub file size limits, the data files and pretrained model files (`.joblib` files) are not included in this repository. To reproduce the results, run the full pipeline (`python scripts/run_full_pipeline.py`) which will download the data and train the models. The `.gitignore` file excludes these large artifacts from version control.

### Module Highlights
- `src.config`: central paths, random seed, split ratio, and hyperparameter grids.
- `src.data_download`: pulls only entries with complete elasticity tensors via `MPRester`.
- `src.preprocessing`: removes entries with missing tensors, computes VRH Young’s modulus (using `pymatgen`’s `ElasticTensor` when needed), filters outliers, and deduplicates.
- `src.features`: applies matminer `ElementFraction` + `ElementProperty (Magpie)` featurizers, encodes crystal system & space-group bins, and engineers aggregate compositional statistics.
- `src.train`: performs the 80/20 split, runs `RandomizedSearchCV` with 5-fold CV for both models, and saves the best estimators alongside CV results and cached splits.
- `src.evaluate`: reloads the cached test set, measures MAE/RMSE/R² (overall + per crystal system), stores metrics, and leverages `src.visualization` for diagnostic plots and RF feature importances.
- `src.visualization`: Matplotlib/seaborn helpers for distribution, scatter, residual, correlation, and importance plots.
- `src.utils`: logging, seeding, environment loading, JSON serialization, and path utilities.
- `scripts/run_*`: thin orchestration wrappers to simplify reproducible CLI execution.

With dependencies installed and `MP_API_KEY` provided, running `python scripts/run_full_pipeline.py` will regenerate every artifact and quantify how close composition + symmetry descriptors can get to high-fidelity elastic modulus predictions without explicit structural models.

---

### Results Summary

#### Model Performance

**Random Forest Regressor:**
- **Test Set Performance**: MAE = 23.80 GPa, RMSE = 38.22 GPa, R² = 0.836
- **Training Set Performance**: MAE = 11.43 GPa, RMSE = 21.25 GPa, R² = 0.949
- **Cross-Validation**: Best CV MAE = 25.37 GPa (std = 1.27 GPa)
- **Best Hyperparameters**: n_estimators=600, max_depth=25, min_samples_split=2, min_samples_leaf=2, max_features=0.6

**Multilayer Perceptron (MLP):**
- **Test Set Performance**: MAE = 27.73 GPa, RMSE = 44.63 GPa, R² = 0.776
- **Training Set Performance**: MAE = 11.72 GPa, RMSE = 19.57 GPa, R² = 0.957
- **Cross-Validation**: Best CV MAE = 28.41 GPa (std = 1.10 GPa)
- **Best Hyperparameters**: hidden_layer_sizes=(256, 128, 64), activation='relu', alpha=0.01, learning_rate_init=0.005

#### Key Findings

1. **Random Forest outperforms MLP** on the test set, achieving ~4 GPa lower MAE and ~0.06 higher R²
2. **Feature Importance Analysis** (Random Forest):
   - **Magpie Statistical Features**: 96.9% of total importance (dominant contributor)
     - Top feature: `MagpieData mean GSvolume_pa` (21.6% importance)
   - **Engineered Features**: 1.2% of total importance
     - Top feature: `avg_valence_electrons` (0.8% importance)
   - **Symmetry Features**: 1.8% of total importance
     - Top feature: `sg_201_230` (high symmetry space groups, 0.4% importance)
   - **ElementFraction Features**: <0.1% of total importance (negligible)
3. **Model Generalization**: Both models show a train-test gap (train R² ~0.95 vs test R² ~0.78-0.84), indicating some overfitting, though Random Forest generalizes better
4. **Prediction Accuracy**: Most predictions fall within ±25-30 GPa of true values for Random Forest, suitable for materials screening applications

All evaluation metrics, diagnostic plots (prediction scatter plots, residual distributions, learning curves, feature importances), and per-crystal-system performance breakdowns are saved in `data/processed/figures/` and `models/evaluation_metrics.joblib`.

---

### AI-Assisted Development Attribution

This project utilized AI-assisted code generation through Cursor (an AI-powered code editor) to accelerate development. Many Python source files in this repository were created by prompting Cursor with specific requirements, and each file includes header comments documenting the prompt used to generate it.

**Key Points:**
- **Code Generation**: All source code files were generated using Cursor AI based on detailed prompts describing the desired functionality
- **Design Decisions**: All design choices (feature engineering strategies, model architectures, hyperparameter grids, preprocessing approaches) were made by Jay Parmar
- **Hyperparameter Tuning**: Model hyperparameter tuning and selection was performed by Jay Parmar
- **Code Review & Validation**: All AI-generated code was reviewed, tested, and validated to ensure correctness and alignment with project requirements
- **Transparency**: Each Python file contains header comments indicating it was created via Cursor prompting, with the specific prompt documented

The use of AI-assisted development tools significantly accelerated the implementation timeline while maintaining full control over design decisions and ensuring code quality through thorough review and testing.



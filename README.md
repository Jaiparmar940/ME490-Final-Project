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

### AI-Assisted Development Attribution

This project utilized AI-assisted code generation through Cursor (an AI-powered code editor) to accelerate development. All Python source files in this repository were created by prompting Cursor with specific requirements, and each file includes header comments documenting the prompt used to generate it.

**Key Points:**
- **Code Generation**: All source code files were generated using Cursor AI based on detailed prompts describing the desired functionality
- **Design Decisions**: All design choices (feature engineering strategies, model architectures, hyperparameter grids, preprocessing approaches) were made by Jay Parmar
- **Hyperparameter Tuning**: Model hyperparameter tuning and selection was performed by Jay Parmar
- **Code Review & Validation**: All AI-generated code was reviewed, tested, and validated to ensure correctness and alignment with project requirements
- **Transparency**: Each Python file contains header comments indicating it was created via Cursor prompting, with the specific prompt documented

The use of AI-assisted development tools significantly accelerated the implementation timeline while maintaining full control over design decisions and ensuring code quality through thorough review and testing.



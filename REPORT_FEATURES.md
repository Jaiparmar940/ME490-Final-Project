# Feature Engineering Report Sections

## \subsubsection{Compositional Features}

Compositional features capture the elemental makeup of materials without requiring explicit structural information. Two primary featurization approaches were employed using the matminer library:

### Element Fraction Features
The `ElementFraction` featurizer generates binary features representing the atomic fraction of each element present in the composition. For a material with formula $A_x B_y C_z$, the featurizer produces features such as $\text{ElementFraction}_A = x/(x+y+z)$, $\text{ElementFraction}_B = y/(x+y+z)$, etc. These features provide a normalized representation of elemental abundance, ensuring that compositions with different total atom counts are directly comparable. The feature set includes one column per unique element encountered across the entire dataset, resulting in a sparse representation where most entries are zero for elements not present in a given material.

### Magpie Statistical Features
The `ElementProperty` featurizer with the "magpie" preset computes statistical descriptors (mean, maximum, minimum, range, mode, and standard deviation) across elemental properties for all elements in a composition. The Magpie dataset includes 132 elemental properties such as:
- Atomic properties: atomic number, atomic radius, electronegativity, ionization energy
- Electronic properties: number of valence electrons, electron affinity, band gap
- Thermodynamic properties: melting point, boiling point, heat of fusion
- Structural properties: density, molar volume
- Other descriptors: covalent radius, ionic radius, etc.

For each property, six statistical moments are computed (mean, max, min, range, mode, std), resulting in approximately 792 features (132 properties × 6 statistics). These features capture how elemental properties are distributed within a composition, providing rich information about the material's expected behavior based on its constituent elements.

**Rationale**: Compositional features are fundamental because elastic modulus is strongly influenced by the types and proportions of elements present. Element fractions directly encode stoichiometry, while Magpie statistics aggregate elemental property information in ways that correlate with bulk material properties like stiffness.

---

## \subsubsection{Symmetry Features}

Symmetry features encode crystallographic information that influences elastic properties through the material's structural organization, without requiring explicit atomic coordinates or unit cell parameters.

### Crystal System Encoding
Crystal systems (cubic, hexagonal, tetragonal, orthorhombic, monoclinic, triclinic, and trigonal) were one-hot encoded to create binary indicator features (e.g., `crys_cubic`, `crys_hexagonal`, etc.). The crystal system fundamentally determines the symmetry constraints on the elastic tensor, with cubic systems having the highest symmetry (3 independent elastic constants) and triclinic systems having the lowest (21 independent constants). This encoding allows the model to learn system-specific relationships between composition and elastic modulus.

### Space Group Binning
Space group numbers (ranging from 1 to 230) were binned into five categories to reduce dimensionality while preserving meaningful symmetry distinctions:
- `sg_1_50`: Space groups 1-50 (low symmetry, including triclinic, monoclinic, and some orthorhombic)
- `sg_51_100`: Space groups 51-100 (moderate symmetry, primarily orthorhombic and tetragonal)
- `sg_101_150`: Space groups 101-150 (higher symmetry, including tetragonal and hexagonal)
- `sg_151_200`: Space groups 151-200 (high symmetry, including hexagonal and cubic)
- `sg_201_230`: Space groups 201-230 (highest symmetry, primarily cubic)

These bins were one-hot encoded to create binary features. Binning was chosen over direct space group encoding to avoid creating 230 sparse features while still capturing the broad symmetry trends that correlate with elastic anisotropy and modulus values.

**Rationale**: Symmetry features are critical because the elastic tensor's structure depends on point group symmetry. Materials with higher symmetry (e.g., cubic) often exhibit different modulus values and anisotropy patterns compared to lower-symmetry systems, even for similar compositions. The space group bins provide a coarse-grained representation of these symmetry effects.

---

## \subsubsection{Engineered Features (optional)}

In addition to the matminer-generated features, three simple heuristic descriptors were manually engineered from the composition objects to capture aggregate compositional properties:

### Mean Atomic Number
Computed as the weighted average of atomic numbers ($Z$) across all elements in the composition:
$$\text{mean\_atomic\_number} = \sum_i f_i \cdot Z_i$$
where $f_i$ is the atomic fraction of element $i$ and $Z_i$ is its atomic number. This feature correlates with average atomic mass and provides a simple proxy for material density and bonding strength.

### Maximum Atomic Number
The maximum atomic number among all elements in the composition:
$$\text{max\_atomic\_number} = \max_i Z_i$$
This captures the presence of heavy elements, which can significantly influence bulk modulus and overall material stiffness.

### Average Valence Electrons
Computed as the weighted average of the most common oxidation state for each element:
$$\text{avg\_valence\_electrons} = \sum_i f_i \cdot \text{oxidation\_state}_i$$
where the oxidation state is taken as the first entry in each element's `common_oxidation_states` list. This feature approximates the average number of valence electrons available for bonding, which relates to bond strength and material cohesion.

**Rationale**: These engineered features provide compact, interpretable summaries of compositional characteristics that may not be fully captured by the statistical aggregation in Magpie features. They serve as simple baselines and can help the model quickly identify materials with extreme compositional properties (e.g., very heavy elements or unusual valence configurations) that might require special treatment.

**Note**: While these features are optional in the sense that the model could potentially learn similar relationships from the Magpie statistics, they provide explicit, interpretable descriptors that may improve model performance and explainability, particularly for tree-based models like Random Forest that can directly utilize these simple heuristics.

---

# Machine Learning Pipeline Report Sections

## \section{4.1 Machine Learning Pipeline Overview}

The machine learning pipeline follows a modular, end-to-end workflow designed to predict elastic modulus from compositional and symmetry features. The pipeline consists of five sequential stages:

1. **Data Acquisition**: Raw elasticity data is downloaded from the Materials Project database via the MPRester API, filtering for entries with complete elastic tensor information.

2. **Data Preprocessing**: The raw dataset undergoes cleaning operations including:
   - Computation of Voigt-Reuss-Hill (VRH) averaged Young's modulus from elastic tensor components
   - Removal of entries with missing or invalid modulus values
   - Deduplication based on material identifiers
   - Outlier filtering using percentile-based clipping (1st to 99th percentile)

3. **Feature Engineering**: Compositional and symmetry features are generated using matminer featurizers and custom engineering functions, producing a high-dimensional feature matrix (approximately 800+ features) from composition objects and crystallographic metadata.

4. **Model Training and Tuning**: Two regression models (Random Forest and Multilayer Perceptron) are trained using an 80/20 train/test split. Hyperparameter optimization is performed via randomized search with 5-fold cross-validation on the training set.

5. **Model Evaluation**: Trained models are evaluated on the held-out test set using MAE, RMSE, and R² metrics. Additional analysis includes per-crystal-system performance breakdowns, prediction vs. true value scatter plots, residual distributions, and feature importance visualizations (for Random Forest).

The pipeline is implemented as a series of modular Python scripts that can be executed independently or as a complete end-to-end workflow. All intermediate artifacts (cleaned data, feature matrices, trained models, evaluation metrics) are persisted to disk to enable reproducibility and incremental development.

---

## \section{4.2 Model Selection Rationale}

Two fundamentally different machine learning algorithms were selected to provide complementary approaches to the regression problem:

### Ensemble Tree-Based Model (Random Forest)
Random Forest was chosen for its ability to:
- Handle high-dimensional, mixed-type feature spaces without requiring feature scaling
- Provide intrinsic feature importance rankings for interpretability
- Capture non-linear relationships and feature interactions through tree-based splitting
- Offer robust performance with minimal hyperparameter tuning
- Naturally handle sparse features (e.g., element fractions where most values are zero)

### Neural Network Model (Multilayer Perceptron)
The MLP was selected to:
- Learn complex, non-linear mappings between high-dimensional feature spaces and the target
- Potentially capture subtle compositional and symmetry interactions that tree models might miss
- Serve as a baseline for more sophisticated deep learning approaches
- Benefit from feature scaling to improve convergence and performance

**Comparative Rationale**: By comparing these two model families, we can assess whether the problem benefits from the explicit feature interactions captured by trees or the learned representations of neural networks. Random Forest provides interpretability through feature importances, while MLP offers flexibility for capturing complex patterns. Both models are well-established in materials informatics and provide strong baselines for elastic property prediction.

---

## \subsection{4.2.1 Random Forest Regressor}

The Random Forest Regressor is an ensemble method that aggregates predictions from multiple decision trees, each trained on a bootstrap sample of the data with random feature subsets at each split.

### Architecture
- **Base Estimators**: 400 decision trees (n_estimators=400 in base configuration)
- **Bootstrap Sampling**: Each tree is trained on a random sample (with replacement) of the training data
- **Feature Subsampling**: At each split, a random subset of features is considered (controlled by `max_features` parameter)
- **Aggregation**: Final prediction is the mean of all tree predictions

### Key Hyperparameters Tuned
- `n_estimators`: Number of trees in the ensemble (200, 400, 600)
- `max_depth`: Maximum depth of trees (None for unlimited, 15, 25)
- `min_samples_split`: Minimum samples required to split a node (2, 5, 10)
- `min_samples_leaf`: Minimum samples required in a leaf node (1, 2, 4)
- `max_features`: Number of features to consider at each split ("sqrt", "log2", 0.6)

### Advantages for This Problem
- **No Feature Scaling Required**: Tree-based models are invariant to feature scaling, simplifying the pipeline
- **Feature Importance**: Provides interpretable rankings of which features (element fractions, Magpie statistics, symmetry indicators) most strongly influence elastic modulus predictions
- **Handles Sparse Features**: Efficiently processes sparse element fraction vectors where most entries are zero
- **Robust to Outliers**: Tree-based splitting is less sensitive to extreme values compared to distance-based methods

### Implementation Details
The Random Forest pipeline consists of a single step containing the `RandomForestRegressor` with `random_state=42` for reproducibility. No preprocessing steps (e.g., scaling, imputation) are required, as the model handles missing values and mixed feature scales natively.

---

## \subsection{4.2.2 Multilayer Perceptron (Neural Network)}

The Multilayer Perceptron (MLP) is a feedforward neural network that learns non-linear mappings through multiple layers of interconnected neurons with non-linear activation functions.

### Architecture
- **Input Layer**: Receives the full feature vector (800+ features)
- **Hidden Layers**: Two or three fully connected layers with ReLU or tanh activation
  - Base configuration: (256, 128) neurons per layer
  - Alternative configurations: (128, 128), (256, 128, 64)
- **Output Layer**: Single neuron with linear activation for regression
- **Optimization**: Adam optimizer with adaptive learning rate

### Key Hyperparameters Tuned
- `hidden_layer_sizes`: Architecture of hidden layers ((128, 128), (256, 128), (256, 128, 64))
- `activation`: Non-linear activation function ("relu", "tanh")
- `alpha`: L2 regularization strength (1e-4, 1e-3, 1e-2)
- `learning_rate_init`: Initial learning rate for Adam optimizer (1e-3, 5e-3)
- `max_iter`: Maximum iterations for training (1000, fixed)

### Preprocessing Pipeline
The MLP requires feature scaling due to the sensitivity of gradient-based optimization to feature scales. A `StandardScaler` is applied before the model, transforming features to zero mean and unit variance:
$$z_i = \frac{x_i - \mu_i}{\sigma_i}$$
where $\mu_i$ and $\sigma_i$ are the mean and standard deviation of feature $i$ computed on the training set.

### Advantages for This Problem
- **Non-Linear Representations**: Can learn complex interactions between compositional and symmetry features that may not be captured by tree-based models
- **Scalability**: Can handle high-dimensional feature spaces efficiently
- **Flexibility**: Architecture can be adapted to capture hierarchical feature relationships

### Implementation Details
The MLP pipeline consists of two steps: (1) `StandardScaler` for feature normalization, and (2) `MLPRegressor` with `random_state=42` for weight initialization reproducibility. The model uses early stopping implicitly through the `max_iter` parameter and L2 regularization to prevent overfitting.

---

## \section{4.3 Data Splitting and Cross-Validation Strategy}

### Train/Test Split
The dataset is partitioned into training and test sets using an 80/20 split (`test_size=0.2`) with a fixed random seed (`random_state=42`) to ensure reproducibility. The split is performed using scikit-learn's `train_test_split` function, which performs stratified random sampling to maintain similar distributions of the target variable across splits.

**Rationale for 80/20 Split**: This ratio provides sufficient training data (~8,000 samples) for learning complex patterns while reserving a substantial test set (~2,000 samples) for reliable performance estimation. The test set is held out completely during model training and hyperparameter tuning to provide an unbiased estimate of generalization performance.

### Split Persistence
To ensure consistent evaluation across multiple runs, the train/test split indices are serialized to `train_test_split_indices.json`. Additionally, the test set (features and targets) is cached as `test_set.joblib`. This persistence mechanism allows:
- Reproducible evaluation: The same test samples are used for all model comparisons
- Incremental development: New models can be evaluated on the identical test set without re-running the entire pipeline
- Consistency: Prevents data leakage that could occur if splits were regenerated with different random states

### Cross-Validation Strategy
Hyperparameter tuning employs **5-fold cross-validation** (`cv=5`) on the training set. The training data is divided into 5 folds, and for each hyperparameter combination:
1. The model is trained on 4 folds
2. Performance is evaluated on the held-out fold
3. This process repeats 5 times, with each fold serving as the validation set once
4. The mean validation score across all 5 folds is computed

**Scoring Metric**: Cross-validation uses negative mean absolute error (neg_MAE) as the scoring metric, which is maximized during search. The best hyperparameters are those that minimize MAE on average across the 5 folds.

**Rationale for 5-Fold CV**: Five folds provide a good balance between:
- Computational efficiency: More folds increase training time
- Statistical reliability: Fewer folds (e.g., 3) may provide less stable performance estimates
- Data utilization: Each fold contains ~20% of training data, providing robust validation while maximizing training data per fold

The cross-validation strategy ensures that hyperparameter selection is based on generalization performance rather than training set performance, reducing overfitting risk.

---

## \section{4.4 Hyperparameter Tuning Procedures}

Hyperparameter optimization is performed using **RandomizedSearchCV**, which randomly samples from the specified parameter grids rather than exhaustively searching all combinations.

### Randomized Search Strategy
For each model, a hyperparameter grid is defined with discrete values or ranges for each parameter. RandomizedSearchCV:
1. Randomly samples `n_iter` combinations from the grid (where `n_iter` is the minimum of 20 and the total grid size)
2. Evaluates each combination using 5-fold cross-validation
3. Selects the combination with the best average validation score
4. Refits the model on the full training set with the best parameters

**Advantages of Randomized Search**:
- **Computational Efficiency**: Evaluates only a subset of possible combinations, making it feasible for large parameter grids
- **Exploration**: Can discover good hyperparameters even when the grid is large
- **Flexibility**: Easy to add or remove parameter options without exponential growth in search space

### Random Forest Hyperparameter Grid
```python
{
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [None, 15, 25],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.6]
}
```
**Total Combinations**: 3 × 3 × 3 × 3 × 3 = 243 combinations
**Search Budget**: min(20, 243) = 20 random samples

### MLP Hyperparameter Grid
```python
{
    "model__hidden_layer_sizes": [(128, 128), (256, 128), (256, 128, 64)],
    "model__alpha": [1e-4, 1e-3, 1e-2],
    "model__learning_rate_init": [1e-3, 5e-3],
    "model__activation": ["relu", "tanh"]
}
```
**Total Combinations**: 3 × 3 × 2 × 2 = 36 combinations
**Search Budget**: min(20, 36) = 20 random samples

### Search Configuration
- **Random State**: `random_state=42` ensures reproducible hyperparameter sampling
- **Parallelization**: `n_jobs=-1` utilizes all available CPU cores for parallel cross-validation
- **Refitting**: `refit=True` automatically retrains the best model on the full training set after search completion
- **Verbosity**: `verbose=1` provides progress updates during search

### Results Persistence
For each model, the complete cross-validation results (all parameter combinations and their scores) are saved to CSV files (`random_forest_cv_results.csv`, `mlp_cv_results.csv`). This enables:
- Post-hoc analysis of hyperparameter sensitivity
- Identification of near-optimal parameter regions
- Documentation of the tuning process for reproducibility

The best hyperparameters and corresponding cross-validation MAE are logged and stored alongside the trained model artifacts.

---

## \section{4.5 Implementation Details (Libraries, Versions, Reproducibility)}

### Core Libraries and Versions
The project relies on the following key Python packages:

**Machine Learning and Data Processing**:
- `scikit-learn`: Model training, cross-validation, preprocessing, and evaluation metrics
- `numpy<2.0.0`: Numerical computations and array operations
- `pandas`: Data manipulation, CSV/Parquet I/O, and DataFrame operations
- `joblib`: Model serialization and parallel processing

**Materials Science**:
- `pymatgen`: Composition parsing, crystal structure analysis, and elastic tensor computations
- `matminer`: Feature engineering (ElementFraction, ElementProperty/Magpie featurizers)
- `mp-api`: Materials Project API client for data acquisition

**Visualization and Utilities**:
- `matplotlib`: Plotting and figure generation
- `seaborn`: Statistical visualization and enhanced plot styling
- `python-dotenv`: Environment variable management for API keys
- `pyarrow`: Parquet file format support for efficient feature matrix storage
- `tqdm`: Progress bars for long-running operations

**Note**: Specific version numbers are not pinned in `requirements.txt` to allow flexibility with dependency resolution, but `numpy<2.0.0` is constrained to ensure compatibility with other packages. For full reproducibility, consider generating a `requirements-lock.txt` using `pip freeze` after installation.

### Reproducibility Measures

#### Random Seed Management
A centralized random seed (`RANDOM_SEED=42`) is defined in `src/config.py` and applied consistently across:
- **Python's random module**: `random.seed(42)`
- **NumPy random number generator**: `np.random.seed(42)`
- **Hash randomization**: `PYTHONHASHSEED=42` environment variable
- **Scikit-learn operations**: `random_state=42` for train_test_split, RandomizedSearchCV, RandomForestRegressor, and MLPRegressor

This ensures that:
- Data splits are identical across runs
- Hyperparameter search samples the same combinations
- Model initializations (neural network weights, tree bootstrapping) are deterministic
- Results are fully reproducible

#### Deterministic Data Splits
The train/test split is performed once with a fixed random seed and persisted to disk:
- Split indices are saved to `train_test_split_indices.json`
- Test set features and targets are cached in `test_set.joblib`
- Evaluation always uses the same held-out samples

#### Artifact Caching
Intermediate pipeline artifacts are cached to enable:
- **Incremental development**: Skip completed stages (e.g., feature engineering) when only downstream components change
- **Reproducibility**: Reuse exact same feature matrices and splits across experiments
- **Efficiency**: Avoid redundant computations (e.g., re-downloading data, re-featurizing)

Cached artifacts include:
- Raw data: `data/raw/elasticity_raw.csv`
- Cleaned data: `data/processed/elasticity_cleaned.csv`
- Feature matrix: `data/processed/elasticity_features.parquet`
- Train/test split: `data/processed/train_test_split_indices.json` and `test_set.joblib`
- Trained models: `models/random_forest.joblib`, `models/mlp.joblib`
- Cross-validation results: `models/*_cv_results.csv`
- Evaluation metrics: `models/evaluation_metrics.joblib`

#### Code Organization
The codebase follows a modular structure with clear separation of concerns:
- **Configuration**: Centralized in `src/config.py` (paths, seeds, hyperparameters)
- **Data pipeline**: Separate modules for download, preprocessing, and feature engineering
- **Modeling**: Isolated model definitions, training, and evaluation logic
- **Utilities**: Shared functions for logging, reproducibility, and file I/O

This organization facilitates:
- **Maintainability**: Changes to one component don't affect others
- **Testability**: Individual modules can be tested in isolation
- **Reproducibility**: Clear data flow and dependency chain

#### Execution Workflow
The pipeline can be executed via:
1. **Individual scripts**: Run each stage separately (`run_download.py`, `run_train.py`, `run_evaluate.py`)
2. **Full pipeline**: Single command execution (`run_full_pipeline.py`) that orchestrates all stages
3. **Command-line arguments**: Flexible input/output path specification for custom workflows

All scripts use the same configuration and random seeds, ensuring consistent results regardless of execution method.

### Environment Setup
Reproducibility requires:
1. **Python environment**: Virtual environment (`.venv`) to isolate dependencies
2. **API key**: Materials Project API key set as `MP_API_KEY` environment variable (loaded via `python-dotenv`)
3. **Directory structure**: Automatic creation of required directories (`data/raw/`, `data/processed/`, `models/`)

The `README.md` provides step-by-step setup instructions to recreate the exact environment used for the project.

---

# Model Training and Validation Report Sections

## \section{5.1 Training Procedure}

The training procedure follows a systematic workflow that integrates hyperparameter tuning with model training, ensuring optimal model selection while maintaining strict separation between training and test data.

### Initialization and Data Loading
1. **Random Seed Initialization**: The training process begins by setting a global random seed (`RANDOM_SEED=42`) using `set_seed()`, which seeds Python's random module, NumPy's random number generator, and sets the `PYTHONHASHSEED` environment variable. This ensures complete reproducibility of all random operations.

2. **Feature Matrix Loading**: The feature matrix and target vector are loaded from the persisted Parquet file (`elasticity_features.parquet`). The dataset is split into feature matrix $X$ (containing all compositional, symmetry, and engineered features) and target vector $y$ (VRH Young's modulus values).

3. **Train/Test Split**: The dataset is partitioned using `train_test_split()` with `test_size=0.2` and `random_state=42`. This creates:
   - Training set: 80% of samples (~8,000 materials) used for model training and hyperparameter tuning
   - Test set: 20% of samples (~2,000 materials) held out completely for final evaluation

4. **Split Persistence**: The train and test indices are serialized to `train_test_split_indices.json`, and the test set is cached to `test_set.joblib` to ensure consistent evaluation across multiple runs.

### Model Training Workflow
For each model (Random Forest and MLP), the following procedure is executed:

1. **Pipeline Initialization**: The base model pipeline is created using factory functions (`create_random_forest_pipeline()` or `create_mlp_pipeline()`). These pipelines include:
   - Random Forest: Single-step pipeline with `RandomForestRegressor`
   - MLP: Two-step pipeline with `StandardScaler` followed by `MLPRegressor`

2. **Hyperparameter Search**: `RandomizedSearchCV` is configured with:
   - Parameter grid: Model-specific hyperparameter distributions
   - Number of iterations: `min(20, total_grid_size)` to balance exploration and computational cost
   - Cross-validation: 5-fold CV on the training set
   - Scoring metric: Negative mean absolute error (neg_MAE)
   - Parallelization: `n_jobs=-1` to utilize all available CPU cores
   - Random state: Fixed seed for reproducible hyperparameter sampling

3. **Cross-Validation Execution**: For each hyperparameter combination:
   - The training set is divided into 5 folds
   - The model is trained on 4 folds and validated on the held-out fold
   - This process repeats 5 times (each fold serves as validation once)
   - The mean validation MAE across all 5 folds is computed

4. **Best Model Selection**: The hyperparameter combination with the lowest mean validation MAE (highest neg_MAE score) is selected. The `refit=True` parameter ensures the best model is automatically retrained on the entire training set.

5. **Model Persistence**: The best estimator (trained on full training set) is serialized using `joblib.dump()` to:
   - `models/random_forest.joblib` for Random Forest
   - `models/mlp.joblib` for MLP

6. **Results Documentation**: Complete cross-validation results are saved to CSV files (`*_cv_results.csv`), containing:
   - All tested hyperparameter combinations
   - Individual fold scores (`split0_test_score` through `split4_test_score`)
   - Mean and standard deviation of CV scores
   - Rank of each combination
   - Fit and score times for performance analysis

### Training Characteristics

**Random Forest Training**:
- Ensemble method: Aggregates predictions from 400 base decision trees (in base configuration)
- Bootstrap sampling: Each tree trained on random sample with replacement
- Feature subsampling: Random feature subset considered at each split
- No preprocessing required: Handles raw feature values directly

**MLP Training**:
- Feature scaling: `StandardScaler` normalizes features to zero mean and unit variance before training
- Gradient-based optimization: Adam optimizer with adaptive learning rates
- Iterative training: Up to 1000 iterations with early stopping potential
- Regularization: L2 penalty (alpha parameter) prevents overfitting

### Computational Considerations
- **Parallelization**: Cross-validation folds are evaluated in parallel across CPU cores
- **Memory Management**: Models and intermediate results are persisted to disk to avoid memory issues
- **Caching**: Test set and split metadata are cached to avoid redundant computations
- **Progress Tracking**: Verbose logging (`verbose=1`) provides real-time updates during hyperparameter search

The training procedure is designed to be fully automated, reproducible, and efficient, with all intermediate results and final models saved for subsequent evaluation and analysis.

---

## \section{5.2 Validation Metrics}

Model performance is assessed using three complementary regression metrics that capture different aspects of prediction accuracy: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).

### Primary Metrics

#### Mean Absolute Error (MAE)
MAE measures the average magnitude of prediction errors without considering their direction:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
where $y_i$ is the true value, $\hat{y}_i$ is the predicted value, and $n$ is the number of samples.

**Properties**:
- Units: Same as target variable (GPa for Young's modulus)
- Robustness: Less sensitive to outliers than RMSE
- Interpretation: Average prediction error in GPa
- Usage: Primary metric for hyperparameter tuning (minimized during cross-validation)

#### Root Mean Squared Error (RMSE)
RMSE penalizes larger errors more heavily than smaller ones:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Properties**:
- Units: Same as target variable (GPa)
- Sensitivity: More sensitive to outliers than MAE
- Interpretation: Standard deviation of prediction errors
- Usage: Provides insight into worst-case prediction performance

#### Coefficient of Determination (R²)
R² measures the proportion of variance in the target variable explained by the model:
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$
where $\bar{y}$ is the mean of true values.

**Properties**:
- Range: $(-\infty, 1]$ (1 = perfect predictions, 0 = no better than mean, negative = worse than mean)
- Unitless: Normalized metric independent of target scale
- Interpretation: Fraction of variance explained (e.g., R² = 0.85 means 85% of variance is explained)
- Usage: Standard metric for regression model comparison

### Metric Computation Contexts

#### Cross-Validation Metrics
During hyperparameter tuning, **negative MAE** (neg_MAE) is used as the scoring function:
- Scikit-learn convention: Higher scores are better, so MAE is negated
- Averaged across 5 folds to provide robust performance estimates
- Used to rank hyperparameter combinations and select the best model

#### Training Set Metrics
After model training, metrics are computed on the full training set to assess:
- **Model fit quality**: How well the model captures patterns in training data
- **Overfitting indicators**: Large gap between train and test metrics suggests overfitting
- **Baseline comparison**: Training metrics provide upper bound on expected test performance

#### Test Set Metrics
The held-out test set provides unbiased estimates of generalization performance:
- **Final evaluation**: Test metrics represent expected performance on new, unseen materials
- **Model comparison**: Enables fair comparison between Random Forest and MLP
- **Performance reporting**: Test metrics are the primary results reported in the study

### Per-Crystal-System Metrics
Additional metrics are computed for each crystal system separately to assess performance heterogeneity:
- **Stratified analysis**: Performance may vary across crystal systems due to different symmetry constraints
- **Minimum sample threshold**: Crystal systems with fewer than 5 test samples are excluded
- **System-specific insights**: Identifies which crystal systems are easier or harder to predict

### Visualization and Diagnostic Metrics
Beyond numerical metrics, several visualizations provide diagnostic insights:

1. **Prediction vs. True Scatter Plots**: 
   - Reveals systematic biases, heteroscedasticity, or non-linear relationships
   - Ideal predictions lie on the $y=x$ diagonal line
   - Deviations indicate model limitations

2. **Residual Histograms**:
   - Distribution of prediction errors ($y_{\text{true}} - y_{\text{pred}}$)
   - Ideal: Normal distribution centered at zero
   - Skewness or multiple modes indicate systematic errors

3. **Feature Importance Plots** (Random Forest only):
   - Identifies which features contribute most to predictions
   - Provides interpretability and feature selection insights
   - Helps validate domain knowledge (e.g., compositional features should be important)

### Metric Interpretation Guidelines
- **MAE < 10 GPa**: Excellent performance for most engineering applications
- **MAE 10-20 GPa**: Good performance, acceptable for screening applications
- **MAE > 20 GPa**: Moderate performance, may require additional features or model improvements
- **R² > 0.8**: Strong predictive capability
- **R² 0.6-0.8**: Moderate predictive capability
- **R² < 0.6**: Weak predictive capability, model may need refinement

These metrics collectively provide a comprehensive assessment of model performance, enabling identification of strengths, weaknesses, and areas for improvement.

---

## \section{5.3 Cross-Validation Results}

Cross-validation results provide detailed insights into hyperparameter sensitivity, model stability, and the robustness of performance estimates across different data subsets.

### Cross-Validation Output Structure
For each model, the complete cross-validation results are saved to CSV files containing comprehensive information about every hyperparameter combination tested:

**Key Columns in CV Results**:
- `mean_test_score`: Mean validation score (negative MAE) across 5 folds
- `std_test_score`: Standard deviation of validation scores across folds
- `split0_test_score` through `split4_test_score`: Individual fold scores
- `rank_test_score`: Ranking of each combination (1 = best)
- `mean_fit_time`, `std_fit_time`: Average and standard deviation of training time per fold
- `mean_score_time`, `std_score_time`: Average and standard deviation of prediction time per fold
- `params`: Dictionary of hyperparameter values for each combination

### Random Forest Cross-Validation Analysis

**Hyperparameter Space Explored**:
- `n_estimators`: [200, 400, 600] trees
- `max_depth`: [None, 15, 25]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ["sqrt", "log2", 0.6]

**Total Combinations**: 243 possible combinations
**Samples Evaluated**: 20 random combinations (8.2% of grid)

**Typical Observations**:
- **Tree depth**: Deeper trees (max_depth=None) often perform well but may overfit; moderate depths (15-25) can provide better generalization
- **Sample constraints**: Higher `min_samples_split` and `min_samples_leaf` values tend to reduce overfitting but may underfit complex patterns
- **Feature subsampling**: `max_features="sqrt"` or `0.6` typically outperform `"log2"` for this high-dimensional feature space
- **Ensemble size**: More trees (400-600) generally improve performance but with diminishing returns and increased computational cost

**Performance Stability**:
- Standard deviation of CV scores (`std_test_score`) indicates consistency across folds
- Low standard deviation (< 1 GPa) suggests stable performance across different data subsets
- High standard deviation may indicate sensitivity to specific data distributions or insufficient training data

### MLP Cross-Validation Analysis

**Hyperparameter Space Explored**:
- `hidden_layer_sizes`: [(128, 128), (256, 128), (256, 128, 64)]
- `activation`: ["relu", "tanh"]
- `alpha`: [1e-4, 1e-3, 1e-2] (L2 regularization strength)
- `learning_rate_init`: [1e-3, 5e-3]

**Total Combinations**: 36 possible combinations
**Samples Evaluated**: 20 random combinations (55.6% of grid)

**Typical Observations**:
- **Architecture**: Deeper networks (256, 128, 64) may capture more complex patterns but require careful regularization
- **Activation functions**: ReLU typically performs well for regression tasks; tanh may provide smoother gradients
- **Regularization**: Higher alpha values (1e-2) prevent overfitting but may underfit; lower values (1e-4) allow more flexibility
- **Learning rate**: Adaptive learning rates in Adam optimizer help, but initial learning rate affects convergence speed and final performance

**Training Characteristics**:
- MLP training times are typically longer than Random Forest due to iterative gradient-based optimization
- Convergence may vary significantly across hyperparameter combinations
- StandardScaler preprocessing is critical for stable training and good performance

### Cross-Validation Score Interpretation

**Mean Test Score**:
- Represents average validation MAE (negated) across 5 folds
- Lower absolute values indicate better performance
- Best model has the highest (least negative) mean test score

**Standard Deviation**:
- Measures consistency of performance across folds
- Low std (< 1 GPa): Stable, reliable model
- High std (> 2 GPa): Unstable performance, may indicate:
  - Insufficient training data
  - High variance in data distribution across folds
  - Overfitting to specific fold characteristics

**Fold-by-Fold Analysis**:
- Individual fold scores (`split0_test_score` through `split4_test_score`) reveal:
  - Consistency: Similar scores across folds indicate robust performance
  - Outliers: One fold with significantly different score may indicate data distribution issues
  - Variance: Large spread between min and max fold scores suggests instability

### Best Hyperparameter Selection

The best hyperparameters are selected based on:
1. **Primary criterion**: Highest mean test score (lowest mean validation MAE)
2. **Secondary considerations**: 
   - Lower standard deviation (more stable)
   - Reasonable computational cost (fit time)
   - Simpler models when performance is similar (Occam's razor)

**Best Model Characteristics**:
- Random Forest: Typically selects moderate tree depth, balanced sample constraints, and efficient feature subsampling
- MLP: Often selects moderate architecture size with appropriate regularization to balance capacity and generalization

### Cross-Validation vs. Test Set Performance

**Expected Relationship**:
- CV scores should be slightly optimistic compared to test set performance (CV uses training data)
- Large discrepancy suggests:
  - Overfitting during hyperparameter tuning
  - Test set distribution differs from training set
  - Insufficient cross-validation folds

**Validation Strategy**:
- CV provides robust hyperparameter selection
- Test set provides final, unbiased performance estimate
- Both metrics are reported to assess model reliability

### Computational Performance

**Training Time Analysis**:
- `mean_fit_time`: Average time to train model on one fold
- Total CV time ≈ `mean_fit_time × n_folds × n_iter`
- Random Forest: Typically faster (seconds to minutes per combination)
- MLP: Typically slower (minutes to tens of minutes per combination)

**Scoring Time Analysis**:
- `mean_score_time`: Average time to make predictions on validation fold
- Important for deployment considerations
- Both models typically have fast prediction times (< 1 second for full test set)

The cross-validation results provide comprehensive insights into model behavior, hyperparameter sensitivity, and performance stability, enabling informed model selection and understanding of model limitations.


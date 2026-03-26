# LIME Stability Under the Variations of Data

A research project of thesis investigates the stability of LIME (Local Interpretable Model-agnostic Explanations) across repeated runs under different conditions of the dataset and models of machine learning.

## Research Questions

- **RQ1** — How stable are LIME explanations across repeated runs for the same input instance?
- **RQ2** — How does the stability of LIME explanations alter in different models of machine learning?
- **RQ3** — How do dataset characteristics such as noise and class imbalance influence the stability of LIME explanations?
- **RQ4** — Under which circumstances can explanations of LIME be considered trustworthy for interpreting the predictions of machine learning?

## Structure of the Project


LIME_stability/
├── lime_stability_vscode.py     # Main experiment — loads data, trains models, runs LIME
├── generate_plots.py            # Generates all 5 result plots from saved CSVs
├── results/
│   ├── tables/                  # CSV result files
│   │   ├── baseline_results.csv
│   │   ├── noise_results.csv
│   │   ├── imbalance_results.csv
│   │   └── summary_results.csv
│   └── figures/                 # PNG plots
│       ├── rq1_baseline_stability.png
│       ├── rq2a_noise.png
│       ├── rq2b_imbalance.png
│       ├── rq3_model_comparison.png
│       └── heatmap_all_conditions.png
└── README.md
```

## Datasets

- UCI Adult Income 
- Breast Cancer Wisconsin 


## Dataset Variations

| Variation | Levels |
|---|---|
| Baseline | No modification |
| Noise |  Gaussian noise (5% and 10%) added to numeric features |
| Class Imbalance | 70/30 and 90/10 majority/minority split |


## Machine Learning Models

| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Random Forest | Ensemble / Non-linear |
| MLP (64→32) | Neural Network |


## Metrics for Stability

**Jaccard Similarity**  measures whether the same features appear in the two explanations of LIME.
**Spearman Rank Correlation**  measures whether features are ranked in the same order in two LIME explanations. 
Both metrics are averaged over all pairwise combinations of 20 LIME runs per test instance, across 30 test instances per condition.


## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/LIME_stability.git
cd LIME_stability
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac or Linux
venv\Scripts\activate           # Windows
```

### 3. Installing the dependencies

```bash
pip install lime scikit-learn numpy pandas matplotlib seaborn scipy certifi
```

### 4. Run the experiment

```bash
python3 lime_stability_vscode.py
```

This will take approximately **50–60 minutes** to complete. Results are saved automatically to `results/tables/`.

### 5. Then generate plots

```bash
python3 generate_plots.py
```

All 5 plots are saved to `results/figures/`.


## Configuration of LIME

| Parameter | Value |
|---|---|
| LIME runs per instance | 20 |
| Test instances per condition | 30 |
| LIME neighbourhood samples | 2000 |
| Top-K features | 5 |
| Discretizer | Quartile |
| Random seed | 42 |


## Dependencies

- Python 3.12
- lime == 0.2.0.1
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- certifi


## Author

Research project for thesis (2026)

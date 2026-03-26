from __future__ import annotations
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import random
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

#configuration 
RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
RAW_DIR = RESULTS_DIR / "raw_explanations"
FIGURES_DIR = RESULTS_DIR / "figures"

for folder in [RESULTS_DIR, TABLES_DIR, RAW_DIR, FIGURES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

global_random_seed = 42
np.random.seed(global_random_seed)
random.seed(global_random_seed)

top_k = 5
n_explain_instances = 30
n_lime_runs = 20
lime_num_samples = 2000

noise_levels = [0.0, 0.05, 0.10]
imbalance_ratios = [0.50, 0.70, 0.90]
test_size = 0.2

#classes of data
@dataclass
class ExperimentConfig:
    dataset_name: str
    variation_type: str
    variation_value: float
    model_name: str
    top_k: int = top_k
    n_explain_instances: int = n_explain_instances
    n_lime_runs: int = n_lime_runs
    lime_num_samples: int = lime_num_samples
    random_seed: int = global_random_seed


@dataclass
class StabilityResult:
    dataset_name: str
    variation_type: str
    variation_value: float
    model_name: str
    mean_jaccard: float
    std_jaccard: float
    mean_spearman: float
    std_spearman: float
    test_accuracy: float
    test_f1: float
    test_roc_auc: float
    n_instances: int
    n_lime_runs: int
    top_k: int

#loaders of dataset
def load_adult_income():
   
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.copy()
    target_col = "class"
    y = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .map({"<=50K": 0, ">50K": 1})
    )
    X = df.drop(columns=[target_col])
    return X, y, "income_over_50k"


def load_breast_cancer_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = pd.Series(data.target, name="target")
    return X, y, "malignant"


def get_dataset(dataset_name: str):
    if dataset_name == "adult_income":
        return load_adult_income()
    elif dataset_name == "breast_cancer":
        return load_breast_cancer_data()
    else:
        raise ValueError(f"This  dataset is unknown: {dataset_name}")


# overview of dataset
def overviews_dataset():
    """prints shape, balance of class, missing values and sample rows."""

    #adult income
    print("=" * 55)
    print("ADULT INCOME DATASET")
    print("=" * 55)
    X_adult, y_adult, _ = load_adult_income()
    df_adult = X_adult.copy()
    df_adult["target"] = y_adult

    print(f"Shape          : {df_adult.shape}")
    print(f"Features       : {X_adult.shape[1]}  |  Samples: {len(df_adult)}")
    print(f"\nClass distribution:")
    print(y_adult.value_counts().rename({0: "<=50K", 1: ">50K"}).to_string())
    print(f"\nClass balance  : {y_adult.value_counts(normalize=True).round(3).to_dict()}")

    missing = df_adult.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"\nMissing values:\n{missing.to_string()}")
    else:
        print("\nMissing values : none")

    num_cols = X_adult.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X_adult.columns if c not in num_cols]
    print(f"\nNumeric features    : {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")
    print(f"  → {list(cat_cols)}")
    print("\nFirst 5 rows:")
    print(df_adult.head().to_string())
    print("\nNumeric statistics:")
    print(X_adult[num_cols].describe().round(2).to_string())

    #breast cancer
    print("\n" + "=" * 55)
    print("BREAST CANCER DATASET")
    print("=" * 55)
    X_bc, y_bc, _ = load_breast_cancer_data()
    df_bc = X_bc.copy()
    df_bc["target"] = y_bc

    print(f"Shape          : {df_bc.shape}")
    print(f"Features       : {X_bc.shape[1]}  |  Samples: {len(df_bc)}")
    print(f"\nClass distribution:")
    print(y_bc.value_counts().rename({0: "malignant", 1: "benign"}).to_string())
    print(f"\nClass balance  : {y_bc.value_counts(normalize=True).round(3).to_dict()}")
    print(f"\nMissing values : {df_bc.isnull().sum().sum()} (none expected)")
    print(f"All features numeric: {X_bc.select_dtypes(include=[np.number]).shape[1] == X_bc.shape[1]}")
    print("\nFirst 5 rows:")
    print(df_bc.head().to_string())
    print("\nNumeric statistics:")
    print(X_bc.describe().round(2).to_string())


overviews_dataset()

#variations of dataset 
def add_numeric_noise(X: pd.DataFrame, noise_level: float, random_seed: int):
    if noise_level == 0:
        return X.copy()
    rng = np.random.default_rng(random_seed)
    X_noisy = X.copy()
    numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        std = X_noisy[col].std(ddof=0)
        if pd.isna(std) or std == 0:
            continue
        noise = rng.normal(loc=0.0, scale=noise_level * std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col].astype(float) + noise
    return X_noisy


def apply_class_imbalance(X: pd.DataFrame, y: pd.Series,
                          majority_ratio: float, random_seed: int):
    value_counts = y.value_counts()
    if len(value_counts) != 2:
        raise ValueError("Just binary classification is supported.")

    majority_class = value_counts.idxmax()
    minority_class = value_counts.idxmin()

    X_major = X[y == majority_class]
    y_major = y[y == majority_class]
    X_minor = X[y == minority_class]
    y_minor = y[y == minority_class]

    n_major = len(X_major)
    desired_n_minor = int(round(n_major * (1 - majority_ratio) / majority_ratio))
    desired_n_minor = max(10, min(desired_n_minor, len(X_minor)))

    X_minor_down, y_minor_down = resample(
        X_minor, y_minor,
        replace=False,
        n_samples=desired_n_minor,
        random_state=random_seed,
    )

    X_out = pd.concat([X_major, X_minor_down], axis=0).sample(
        frac=1, random_state=random_seed)
    y_out = pd.concat([y_major, y_minor_down], axis=0).loc[X_out.index]
    return X_out.reset_index(drop=True), y_out.reset_index(drop=True)

#preprocessor and models
def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(model_name: str, random_seed: int):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=random_seed)
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=random_seed, n_jobs=-1)
    elif model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(X: pd.DataFrame, model_name: str, random_seed: int):
    preprocessor = build_preprocessor(X)
    model = build_model(model_name, random_seed)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        model),
    ])

#metrics of prediction
def compute_prediction_metrics(model: Pipeline,
                                X_test: pd.DataFrame,
                                y_test: pd.Series):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":  roc_auc_score(y_test, y_proba),
    }

#helpers of LIME
def make_lime_explainer(
    X_train_transformed: np.ndarray,
    feature_names: list,
    class_names: list,
    random_seed: int = global_random_seed,
) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        discretizer="quartile",
        random_state=random_seed,
    )


def explain_single_instance_multiple_times(
    explainer: LimeTabularExplainer,
    fitted_pipeline: Pipeline,
    X_test_transformed: np.ndarray,
    instance_index: int,
    n_runs: int,
    top_k: int,
    num_samples: int,
    random_seed: int,
) -> list:
    """
    Run LIME n_runs times on the same instance, each with a different seed.
    Returns a list of (feature_name, weight) tuple lists.
    """
    model        = fitted_pipeline.named_steps["model"]
    instance     = X_test_transformed[instance_index]
    explanations = []

    for run_id in range(n_runs):
        seed = random_seed + run_id
        np.random.seed(seed)
        random.seed(seed)

        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=top_k,
            num_samples=num_samples,
        )

        explanation_list = exp.as_list(label=1)
        parsed = [
            (feature_desc.strip(), float(weight))
            for feature_desc, weight in explanation_list
        ]
        explanations.append(parsed)

    return explanations

#metrics of stability
def explanation_to_feature_set(explanation: list) -> set:
    return {name for name, _ in explanation}


def explanation_to_rank_dict(explanation: list) -> dict:
    ordered = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)
    return {name: rank + 1 for rank, (name, _) in enumerate(ordered)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    union = set_a.union(set_b)
    if not union:
        return 1.0
    return len(set_a.intersection(set_b)) / len(union)


def spearman_rank_similarity(rank_a: dict, rank_b: dict, top_k: int) -> float:
    all_features = sorted(set(rank_a).union(rank_b))
    if len(all_features) < 2:
        return 1.0
    a = [rank_a.get(f, top_k + 1) for f in all_features]
    b = [rank_b.get(f, top_k + 1) for f in all_features]
    corr, _ = spearmanr(a, b)
    return 0.0 if np.isnan(corr) else float(corr)


def compute_stability_for_one_instance(
    explanations: list, top_k: int
) -> tuple[float, float]:
    
    if len(explanations) < 2:
        return 0.0, 0.0

    jaccards, spearmans = [], []
    for exp_a, exp_b in combinations(explanations, 2):
        set_a = explanation_to_feature_set(exp_a)
        set_b = explanation_to_feature_set(exp_b)
        jaccards.append(jaccard_similarity(set_a, set_b))
        rank_a = explanation_to_rank_dict(exp_a)
        rank_b = explanation_to_rank_dict(exp_b)
        spearmans.append(spearman_rank_similarity(rank_a, rank_b, top_k))

    return float(np.mean(jaccards)), float(np.mean(spearmans))

#single experiment runner 
def run_single_experiment(config: ExperimentConfig):
    print(f"\n  dataset={config.dataset_name} | "
          f"variation={config.variation_type}({config.variation_value}) | "
          f"model={config.model_name}")

    X, y, positive_class_name = get_dataset(config.dataset_name)

    #applying variation
    if config.variation_type == "baseline":
        X_var, y_var = X.copy(), y.copy()
    elif config.variation_type == "noise":
        X_var = add_numeric_noise(X, config.variation_value, config.random_seed)
        y_var = y.copy()
    elif config.variation_type == "imbalance":
        X_var, y_var = apply_class_imbalance(
            X, y, config.variation_value, config.random_seed)
    else:
        raise ValueError(f"Unknown variation type: {config.variation_type}")

    # Guard: too few minority samples to stratify
    if y_var.value_counts().min() < 2:
        print("    SKIPPED: too few minority samples to stratify.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_var, y_var,
        test_size=test_size,
        random_state=config.random_seed,
        stratify=y_var,
    )

    #build and fit pipeline
    pipeline = build_pipeline(X_train, config.model_name, config.random_seed)
    pipeline.fit(X_train, y_train)

    pred_metrics = compute_prediction_metrics(pipeline, X_test, y_test)

    preprocessor        = pipeline.named_steps["preprocessor"]
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed  = preprocessor.transform(X_test)
    feature_names       = preprocessor.get_feature_names_out().tolist()

    class_names = ["class_0", positive_class_name]
    explainer   = make_lime_explainer(
        X_train_transformed, feature_names, class_names)

    n_instances    = min(config.n_explain_instances, len(X_test_transformed))
    rng            = np.random.default_rng(config.random_seed)
    chosen_indices = rng.choice(
        len(X_test_transformed), size=n_instances, replace=False)

    instance_rows = []
    all_jaccards  = []
    all_spearmans = []

    for idx in chosen_indices:
        explanations = explain_single_instance_multiple_times(
            explainer=explainer,
            fitted_pipeline=pipeline,
            X_test_transformed=X_test_transformed,
            instance_index=int(idx),
            n_runs=config.n_lime_runs,
            top_k=config.top_k,
            num_samples=config.lime_num_samples,
            random_seed=config.random_seed + int(idx),
        )

        mean_jaccard, mean_spearman = compute_stability_for_one_instance(
            explanations, config.top_k)
        all_jaccards.append(mean_jaccard)
        all_spearmans.append(mean_spearman)

        instance_rows.append({
            "dataset_name":    config.dataset_name,
            "variation_type":  config.variation_type,
            "variation_value": config.variation_value,
            "model_name":      config.model_name,
            "instance_index":  int(idx),
            "mean_jaccard":    mean_jaccard,
            "mean_spearman":   mean_spearman,
        })

    raw_df       = pd.DataFrame(instance_rows)
    raw_filename = (
        f"{config.dataset_name}_{config.variation_type}_"
        f"{str(config.variation_value).replace('.', 'p')}_"
        f"{config.model_name}.csv"
    )
    raw_df.to_csv(RAW_DIR / raw_filename, index=False)

    result = StabilityResult(
        dataset_name    = config.dataset_name,
        variation_type  = config.variation_type,
        variation_value = config.variation_value,
        model_name      = config.model_name,
        mean_jaccard    = float(np.mean(all_jaccards)),
        std_jaccard     = float(np.std(all_jaccards, ddof=1)) if len(all_jaccards) > 1 else 0.0,
        mean_spearman   = float(np.mean(all_spearmans)),
        std_spearman    = float(np.std(all_spearmans, ddof=1)) if len(all_spearmans) > 1 else 0.0,
        test_accuracy   = float(pred_metrics["accuracy"]),
        test_f1         = float(pred_metrics["f1"]),
        test_roc_auc    = float(pred_metrics["roc_auc"]),
        n_instances     = n_instances,
        n_lime_runs     = config.n_lime_runs,
        top_k           = config.top_k,
    )

    
    print(f"    → Jaccard={result.mean_jaccard:.3f} ± {result.std_jaccard:.3f} | "
          f"Spearman={result.mean_spearman:.3f} ± {result.std_spearman:.3f} | "
          f"Acc={result.test_accuracy:.3f}")
    return result

#plan of the experiment
def build_test_experiment_plan():
    return [
        ExperimentConfig(
            dataset_name="breast_cancer",
            variation_type="baseline",
            variation_value=0.0,
            model_name="logistic_regression",
        ),
        ExperimentConfig(
            dataset_name="breast_cancer",
            variation_type="noise",
            variation_value=0.05,
            model_name="logistic_regression",
        ),
    ]


def build_baseline_configs():
    return [
        ExperimentConfig(
            dataset_name=ds, variation_type="baseline",
            variation_value=0.0, model_name=m)
        for ds in ["adult_income", "breast_cancer"]
        for m  in ["logistic_regression", "random_forest", "mlp"]
    ]


def build_noise_configs():
    return [
        ExperimentConfig(
            dataset_name=ds, variation_type="noise",
            variation_value=lvl, model_name=m)
        for ds  in ["adult_income", "breast_cancer"]
        for m   in ["logistic_regression", "random_forest", "mlp"]
        for lvl in [0.05, 0.10]
    ]


def build_imbalance_configs():
    return [
        ExperimentConfig(
            dataset_name=ds, variation_type="imbalance",
            variation_value=ratio, model_name=m)
        for ds    in ["adult_income", "breast_cancer"]
        for m     in ["logistic_regression", "random_forest", "mlp"]
        for ratio in [0.70, 0.90]
    ]

#batch runner
def run_batch(configs: list, label: str) -> pd.DataFrame:
    """Running a list of configs, collect results, save CSV, return DataFrame."""
    print(f"\n{'='*55}")
    print(f"  RUNNING: {label}  ({len(configs)} experiments)")
    print(f"{'='*55}")

    results = []
    for config in configs:
        try:
            result = run_single_experiment(config)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"    FAILED: {config.dataset_name} / "
                  f"{config.variation_type} / {config.model_name}")
            print(f"    Error : {e}")

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(TABLES_DIR / f"{label}_results.csv", index=False)
    print(f"\n  Saved → tables/{label}_results.csv  ({len(df)} rows)")
    return df

#quick test run (2 experiments only)
print("\n" + "="*55)
print("  quick test run!")
print("="*55)
test_results = []
for config in build_test_experiment_plan():
    result = run_single_experiment(config)
    if result is not None:
        test_results.append(result)

test_df = pd.DataFrame([asdict(r) for r in test_results])
print("\nTest results:")
print(test_df.to_string())

#full experiment
baseline_df  = run_batch(build_baseline_configs(),  "baseline")
noise_df     = run_batch(build_noise_configs(),     "noise")
imbalance_df = run_batch(build_imbalance_configs(), "imbalance")

#combineing all the results
baseline_df  = pd.read_csv(TABLES_DIR / "baseline_results.csv")
noise_df     = pd.read_csv(TABLES_DIR / "noise_results.csv")
imbalance_df = pd.read_csv(TABLES_DIR / "imbalance_results.csv")

summary_df = pd.concat([baseline_df, noise_df, imbalance_df], ignore_index=True)
summary_df.to_csv(TABLES_DIR / "summary_results.csv", index=False)
print(f"\nTotal rows in summary: {len(summary_df)}")
print(summary_df.to_string())
print(" Experiment completed successfully. Run: python3 generate_plots.py  to generate plots")

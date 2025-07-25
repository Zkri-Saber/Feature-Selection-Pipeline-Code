import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from thyroid_analysis.data_loader import load_dataset
from thyroid_analysis.preprocessing import apply_knn_imputation, map_diagnostic_group
from thyroid_analysis.modeling import train_and_evaluate_models, get_models, plot_model_comparison, plot_roc_curves

filename = "real_dataset_knn_imputed.csv"
features_path = "results/features/consensus_features.txt"

print(f"📂 Loading dataset: {filename}")
df = load_dataset(filename)
df = apply_knn_imputation(df)
df["Diagnostic Group Code"] = map_diagnostic_group(df["Diagnostic Group Code"])
df = df.dropna(subset=["Diagnostic Group Code"])

with open(features_path, "r") as f:
    selected_features = f.read().splitlines()

X = df[selected_features]
y = df["Diagnostic Group Code"]

models = get_models()
results_df = train_and_evaluate_models(X, y, models)
print("\n✅ Evaluation Results:\n")
print(results_df)

plot_model_comparison(results_df)
plot_roc_curves(X, y, models)
print("📊 Comparison plots saved to results/figures/")
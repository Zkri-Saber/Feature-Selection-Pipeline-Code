# modeling.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=100, solver='lbfgs'),
        "Random Forest": RandomForestClassifier(max_depth=20, min_samples_split=2, n_estimators=500),
        "SVM": SVC(C=10, gamma=1, kernel='rbf', probability=True),
        "XGBoost": XGBClassifier(learning_rate=0.2, max_depth=6, n_estimators=600, use_label_encoder=False, eval_metric="logloss"),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.1, max_depth=7, n_estimators=600),
        "LightGBM": LGBMClassifier(learning_rate=0.1, n_estimators=350, max_depth=8),
        "CatBoost": CatBoostClassifier(depth=8, iterations=500, learning_rate=0.1, verbose=0),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, max_depth=15),
        "AdaBoost": AdaBoostClassifier(learning_rate=0.2, n_estimators=200)
    }

def boruta_feature_selection(X, y):
    forest = RFC(n_estimators=1000, n_jobs=-1, max_depth=5)
    boruta = BorutaPy(estimator=forest, n_estimators='auto', verbose=0, random_state=42)
    boruta.fit(X.values, y.values)
    selected = X.columns[boruta.support_].tolist()
    print(f"✅ Boruta selected features: {selected}")
    return X[selected]

def train_and_evaluate_models(X, y, models):
    results = {}
    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    smote = BorderlineSMOTE(random_state=42)

    for name, model in models.items():
        print(f"🚀 Training {name}...")
        try:
            X_res, y_res = smote.fit_resample(X, y)
            y_pred = cross_val_predict(model, X_res, y_res, cv=skf)
            results[name] = {
                "Accuracy": accuracy_score(y_res, y_pred),
                "Precision": precision_score(y_res, y_pred, average="macro"),
                "Recall": recall_score(y_res, y_pred, average="macro"),
                "F1 Score": f1_score(y_res, y_pred, average="macro"),
                "ROC AUC": roc_auc_score(pd.get_dummies(y_res), pd.get_dummies(y_pred), average="macro", multi_class="ovr"),
            }
        except Exception as e:
            print(f"⚠️ Skipping {name} due to error: {e}")

    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv("results/model_comparison_results.csv")
    print("📄 Saved model comparison results to results/model_comparison_results.csv")
    return results_df

def plot_model_comparison(results_df):
    os.makedirs("results/figures", exist_ok=True)
    results_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Comparison - Evaluation Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/figures/model_comparison.png")
    plt.close()
    print("📊 Comparison plot saved to results/figures/model_comparison.png")

def plot_roc_curves(X, y, models):
    os.makedirs("results/figures", exist_ok=True)
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            smote = BorderlineSMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_scaled, y)

            model.fit(X_res, y_res)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_scaled)
                fpr, tpr, _ = roc_curve(y_bin[:, 0], y_score[:, 0])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        except Exception as e:
            print(f"⚠️ Skipping {name} for ROC due to error: {e}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/roc_curves.png")
    plt.close()
    print("📈 ROC Curves saved to results/figures/roc_curves.png")

def get_stacked_model(models):
    top_estimators = [(name.replace(" ", "_"), model) for name, model in models.items()
                      if name in ["Gradient Boosting", "XGBoost", "LightGBM"]]
    meta_model = LogisticRegression()
    return StackingClassifier(estimators=top_estimators, final_estimator=meta_model, cv=5)

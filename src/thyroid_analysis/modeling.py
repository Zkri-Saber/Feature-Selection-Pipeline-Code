import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver='lbfgs'),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
        "Extra Trees": ExtraTreesClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

def train_and_evaluate_models(X, y, models):
    """
    Train and evaluate multiple models with cross-validation.
    Returns a DataFrame of evaluation results.
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        print(f"🔍 Training {name}...")
        try:
            y_pred = cross_val_predict(model, X, y, cv=skf)
            y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, average='weighted')
            rec = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            auc = roc_auc_score(pd.get_dummies(y), y_proba, average='macro', multi_class='ovr')

            results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC AUC": auc,
            }
        except ValueError as e:
            print(f"⚠️ Skipping {name} due to error: {e}")
            continue

    results_df = pd.DataFrame(results).T  # Convert dict to DataFrame
    results_df.to_csv("results/model_comparison_results.csv", index=True)
    print("📄 Saved comparison results to results/model_comparison_results.csv")
    return results_df


def plot_model_comparison(results):
   
   

    # Save CSV
    os.makedirs("results/metrics", exist_ok=True)
    csv_path = "results/metrics/model_comparison.csv"
    results_df.to_csv(csv_path)
    print(f"📁 Model comparison results saved to {csv_path}")

    # Plot
    results_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Comparison - Evaluation Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/figures/model_comparison.png")
    plt.close()
    print("📊 Comparison plot saved to results/figures/model_comparison.png")

def save_bar_plot_with_labels(x, y, title, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='skyblue')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/{filename}")
    plt.close()
    print(f"📊 Bar plot saved to results/figures/{filename}")
def plot_feature_importance(model, X, y, top_n=10):
    """
    Plot feature importance for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print(f"⚠️ Model {model.__class__.__name__} does not provide feature importances.")
        return

    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    })
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False).head(top_n)

    # Plot
    save_bar_plot_with_labels(
        x=feature_importance_df["Feature"],
        y=feature_importance_df["Importance"],
        title=f"Feature Importance - {model.__class__.__name__}",
        filename=f"feature_importance_{model.__class__.__name__.lower()}.png"
    )

def plot_roc_curves(X, y, models):
    from sklearn.preprocessing import label_binarize

    os.makedirs("results/figures", exist_ok=True)
    y_bin = label_binarize(y, classes=np.unique(y))
    n_classes = y_bin.shape[1]

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        scaler = StandardScaler()
        smote = SMOTE(random_state=42)
        X_scaled = scaler.fit_transform(X)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
        y_bin_resampled = label_binarize(y_resampled, classes=np.unique(y))

        model.fit(X_resampled, y_resampled)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_scaled)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[0], tpr[0], label=f"{name} (AUC = {roc_auc[0]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/roc_curves.png")
    plt.close()
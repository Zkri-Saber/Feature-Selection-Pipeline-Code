import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib_venn import venn3
from collections import Counter


def save_bar_plot(feature_list, title, filename, color=None):
    """
    Save a bar plot of features.
    """
    os.makedirs("results/figures", exist_ok=True)
    feature_counts = Counter(feature_list)
    features, counts = zip(*feature_counts.items())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, counts, color=color)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Count / Importance")
    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}")
    plt.close()


def save_bar_plot_with_labels(features, values, title, filename):
    """
    Save horizontal bar plot with feature labels and values.
    """
    os.makedirs("results/figures", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(features, values, color='skyblue')
    plt.xlabel("Score / Rank / Importance")
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest score on top
    plt.tight_layout()
    plt.savefig(f"results/figures/{filename}")
    plt.close()


def visualize_all_features(F_RFE, F_PCA, F_DT, consensus_features):
    """
    Generate and save visualizations for RFE, PCA, DT, and Consensus.
    """
    save_bar_plot(F_RFE, "RFE Selected Features", "rfe_features.png")
    save_bar_plot(F_PCA, "PCA Top Features", "pca_features.png")
    save_bar_plot(F_DT, "Decision Tree Top Features", "dt_features.png")

    # Consensus voting bar chart
    all_feats = F_RFE + F_PCA + F_DT
    vote_counts = Counter(all_feats)
    sorted_votes = dict(sorted(vote_counts.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_votes.keys(), sorted_votes.values())
    plt.title("Consensus Feature Votes (RFE, PCA, DT)")
    plt.xlabel("Feature")
    plt.ylabel("Votes Received")
    plt.xticks(rotation=45, ha="right")

    for bar, feat in zip(bars, sorted_votes.keys()):
        if vote_counts[feat] >= 2:
            bar.set_color("orange")
        else:
            bar.set_color("gray")

    plt.tight_layout()
    plt.savefig("results/figures/consensus_features.png")
    plt.close()
    print("📊 Bar charts saved to results/figures/")


def plot_feature_venn(F_RFE, F_PCA, F_DT, save_path="results/figures/venn_feature_overlap.png"):
    """
    Plot a Venn diagram comparing features selected by RFE, PCA, and Decision Tree.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    set_rfe = set(F_RFE)
    set_pca = set(F_PCA)
    set_dt = set(F_DT)

    plt.figure(figsize=(8, 6))
    venn3([set_rfe, set_pca, set_dt], set_labels=("RFE", "PCA", "DT"))
    plt.title("Feature Selection Overlap (RFE, PCA, DT)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📸 Venn diagram saved to {save_path}")


def plot_scree(X, save_path="results/figures/scree_plot.png"):
    """
    Generate and save a Scree Plot for PCA.
    """
    pca = PCA()
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_
    components = np.arange(1, len(explained_variance) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(components, explained_variance, marker='o', linestyle='--')
    plt.title('Scree Plot - Explained Variance by Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(components)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Scree plot saved to {save_path}")


def plot_ranked_features_pca(X):
    """
    Rank features by PCA component weight and save bar chart.
    """
    pca = PCA(n_components=len(X.columns))
    pca.fit(X)
    scores = abs(pca.components_[0])
    feature_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    save_bar_plot_with_labels(feature_scores.index, feature_scores.values,
                              "PCA - Feature Ranking by PC1", "pca_ranked_features.png")


def plot_ranked_features_rfe(X, y):
    """
    Rank features by RFE elimination order and save bar chart.
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=1)
    rfe.fit(X, y)

    ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
    save_bar_plot_with_labels(ranking.index, ranking.values,
                              "RFE - Feature Ranking (Lower is Better)", "rfe_ranked_features.png")


def plot_ranked_features_dt(X, y):
    """
    Rank features by Decision Tree importance and save bar chart.
    """
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)

    importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
    save_bar_plot_with_labels(importance.index, importance.values,
                              "Decision Tree - Feature Importances", "dt_ranked_features.png")


def plot_venn_diagram(F_RFE, F_PCA, F_DT):
    """
    Plot and save Venn diagram of feature overlap.
    """
    plt.figure(figsize=(8, 6))
    venn3([set(F_RFE), set(F_PCA), set(F_DT)], set_labels=('RFE', 'PCA', 'DT'))
    plt.title("Feature Selection Overlap (RFE, PCA, DT)")
    os.makedirs("results/figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/figures/venn_feature_overlap.png")
    plt.close()

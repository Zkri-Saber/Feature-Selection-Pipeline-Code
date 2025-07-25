import os
import matplotlib.pyplot as plt
import pandas as pd
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

def visualize_all_features(F_RFE, F_PCA, F_DT, consensus_features):
    """
    Generate and save visualizations for RFE, PCA, DT, and Consensus.
    """

    # Individual visualizations
    save_bar_plot(F_RFE, "RFE Selected Features", "rfe_features.png")
    save_bar_plot(F_PCA, "PCA Top Features", "pca_features.png")
    save_bar_plot(F_DT, "Decision Tree Top Features", "dt_features.png")

    # Combined Voting (Consensus)
    all_feats = F_RFE + F_PCA + F_DT
    vote_counts = Counter(all_feats)

    sorted_votes = dict(sorted(vote_counts.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_votes.keys(), sorted_votes.values())
    plt.title("Consensus Feature Votes (RFE, PCA, DT)")
    plt.xlabel("Feature")
    plt.ylabel("Votes Received")
    plt.xticks(rotation=45, ha="right")

    # Highlight consensus features (votes >= 2) in different color
    for bar, feat in zip(bars, sorted_votes.keys()):
        if vote_counts[feat] >= 2:
            bar.set_color("orange")
        else:
            bar.set_color("gray")

    # Save
    os.makedirs("results/figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/figures/consensus_features.png")
    plt.close()

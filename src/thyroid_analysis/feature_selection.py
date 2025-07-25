
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from collections import Counter

from thyroid_analysis.preprocessing import apply_knn_imputation, map_diagnostic_group
from thyroid_analysis.data_loader import load_dataset
from thyroid_analysis.visualization import (
    visualize_all_features,
    plot_ranked_features_pca,
    plot_ranked_features_rfe,
    plot_ranked_features_dt,
    plot_venn_diagram,
    plot_feature_venn,
    plot_scree
)

def select_by_rfe(X, y, n_features=8):
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)
    return list(X.columns[rfe.support_])

def select_by_pca(X, n_components=8):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    components = abs(pca.components_)
    scores = components.sum(axis=0)
    return list(pd.Series(scores, index=X.columns).sort_values(ascending=False).index[:n_components])

def select_by_decision_tree(X, y, n_features=8):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return list(importances.sort_values(ascending=False).index[:n_features])

def compute_consensus(feature_lists, vote_threshold=2):
    counts = Counter(sum(feature_lists, []))
    return [f for f, c in counts.items() if c >= vote_threshold]

def run_feature_selection(filename: str):
    print("=== Starting Thyroid Disease Feature Selection ===")
    print(f"📂 Loading dataset: {filename}")
    df = load_dataset(filename)
    df = apply_knn_imputation(df)
    df['Diagnostic Group Code'] = map_diagnostic_group(df['Diagnostic Group Code'])

    print("\n✅ Missing values after KNN imputation:")
    print(df.isnull().sum())
    print("\n🧪 Unique values in 'Diagnostic Group Code' BEFORE mapping:")
    print(df['Diagnostic Group Code'].unique())

    df = df.dropna(subset=['Diagnostic Group Code'])
    y = df['Diagnostic Group Code']
    X = df.drop(columns=['Diagnostic Group Code', 'Dx', 'Diagnostic Group'], errors='ignore')
    X = X.select_dtypes(include=['number'])

    print("\n📊 Class distribution after mapping:")
    print(y.value_counts().to_frame(name='count'))

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    selected_rfe = select_by_rfe(X_train, y_train)
    selected_pca = select_by_pca(X_train)
    selected_dt = select_by_decision_tree(X_train, y_train)

    F_final = compute_consensus([selected_rfe, selected_pca, selected_dt], vote_threshold=2)

    print(f"\n🎯 RFE Selected: {selected_rfe}")
    print(f"🔷 PCA Top Features: {selected_pca}")
    print(f"🌲 DT Top Features: {selected_dt}")
    print(f"✅ Consensus Features (>=2 votes): {F_final}")
    visualize_all_features(selected_rfe, selected_pca, selected_dt, F_final)
    plot_feature_venn(selected_rfe, selected_pca, selected_dt)
    print("📊 Feature selection and visualization completed successfully!")

    plot_scree(X)
    print("📈 Scree plot generated and saved.")
    print("=== Feature Selection Completed ===")


    # Step 10: Save ranked diagrams (importance / PCA weights / RFE order)
    print("📊 Generating ranked feature importance plots...")
    plot_ranked_features_rfe(X, y)
    plot_ranked_features_pca(X)
    plot_ranked_features_dt(X, y)
    print("📊 Ranked feature plots saved to results/figures/")

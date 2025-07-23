import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from thyroid_analysis.data_loader import load_raw_data
from thyroid_analysis.preprocessing import (
    apply_knn_imputation,
    map_diagnostic_group,
)

def run_feature_selection(filename: str):
    print("📂 Loading dataset:", filename)
    df = load_raw_data(filename)

    # Apply KNN Imputer (assumes preprocessing removes non-numeric before imputing)
    df_imputed = apply_knn_imputation(df)

    # Display nulls for verification
    print("\n✅ Missing values after KNN imputation:")
    print(df_imputed.isnull().sum())

    # Show class distribution before mapping
    print("\n🧪 Unique values in 'Diagnostic Group Code' BEFORE mapping:")
    print(df_imputed['Diagnostic Group Code'].unique())

    # Map diagnostic labels to 0–3 classes
    df_imputed['Diagnostic Group'] = map_diagnostic_group(df_imputed['Diagnostic Group Code'])

    # Drop rows with unmapped labels (i.e., NaNs)
    df_clean = df_imputed.dropna(subset=['Diagnostic Group'])

    # Confirm mapping results
    print("\n📊 Class distribution after mapping:")
    print(df_clean['Diagnostic Group'].value_counts().rename_axis('Diagnostic Group').to_frame('count'))

    # === Feature selection preparation ===
    numerical_cols = [
        'Age', 'first TSH', 'last TSH', 'first T4', 'last T4',
        'first T3', 'last T3', 'first FT4', 'last FT4', 'first FT3', 'last FT3'
    ]

    X = df_clean[numerical_cols]
    y = df_clean['Diagnostic Group']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === Standard Scaling ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # === RFE ===
    logreg = LogisticRegression(max_iter=5000)
    rfe = RFE(logreg, n_features_to_select=8)
    rfe.fit(X_train_scaled, y_train)
    selected_rfe = [col for col, selected in zip(numerical_cols, rfe.support_) if selected]
    print("\n🎯 RFE Selected:", selected_rfe)

    # === PCA ===
    pca = PCA(n_components=5)
    pca.fit(X_train_scaled)
    pca_features = pd.Series(
        abs(pca.components_[0]),
        index=numerical_cols
    ).nlargest(8).index.tolist()
    print("🔷 PCA Top Features:", pca_features)

    # === Decision Tree Importance ===
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_importance = pd.Series(dt.feature_importances_, index=numerical_cols)
    selected_dt = dt_importance.nlargest(8).index.tolist()
    print("🌲 DT Top Features:", selected_dt)

    # === Consensus Selection (≥2 votes) ===
    all_feats = selected_rfe + selected_dt + pca_features
    consensus = [feat for feat, count in Counter(all_feats).items() if count >= 2]
    print("✅ Consensus Features (>=2 votes):", consensus)

    # Optionally save output
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/consensus_features.txt", "w") as f:
        f.write("\n".join(consensus))

# feature_selection.py

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv('real_dataset_knn_imputed.csv')

# ------------------------------
# 2. Map Diagnostic Group Codes
# ------------------------------
mapping_dict = {
    'No Disease': 0,
    'Hyperthyroidism': 1,
    'Euthyroid': 2,
    'Hypothyroidism': 3
}

target_column = 'Diagnostic Group Code'
df[target_column] = df[target_column].map(mapping_dict)

# Validate mapping
if df[target_column].isnull().any():
    raise ValueError("Some values in Diagnostic Group Code could not be mapped. Please check the dataset.")

# ------------------------------
# 3. Split Features and Target
# ------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]

# ------------------------------
# 4. Recursive Feature Elimination (RFE)
# ------------------------------
print("[INFO] Performing RFE...")
rfe_selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=8)
rfe_selector.fit(X, y)
rfe_features = X.columns[rfe_selector.support_].tolist()
print(f"RFE Selected Features: {rfe_features}")

# ------------------------------
# 5. Principal Component Analysis (PCA)
# ------------------------------
print("[INFO] Performing PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
pca.fit(X_scaled)

pca_feature_set = set()
for comp in pca.components_:
    top_indices = np.argsort(np.abs(comp))[-3:]  # Top 3 features per component
    pca_feature_set.update(X.columns[top_indices])

pca_features = list(pca_feature_set)
print(f"PCA Top Features: {pca_features}")

# ------------------------------
# 6. Decision Tree Feature Importance
# ------------------------------
print("[INFO] Performing Decision Tree Feature Importance...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

importance_scores = dt_model.feature_importances_
top_dt_indices = np.argsort(importance_scores)[-8:]
dt_features = X.columns[top_dt_indices].tolist()
print(f"DT Top Features: {dt_features}")

# ------------------------------
# 7. Consensus Feature Selection (at least 2 methods)
# ------------------------------
print("[INFO] Performing Consensus Feature Selection...")
all_features = rfe_features + pca_features + dt_features
feature_counts = Counter(all_features)
consensus_features = [feat for feat, count in feature_counts.items() if count >= 2]
print(f"Consensus Features (CFS): {consensus_features}")

# ------------------------------
# 8. Save Feature-Selected Datasets
# ------------------------------
print("[INFO] Saving selected features to CSV...")

rfe_df = df[rfe_features + [target_column]]
pca_df = df[pca_features + [target_column]]
dt_df = df[dt_features + [target_column]]
cfs_df = df[consensus_features + [target_column]]

rfe_df.to_csv('selected_rfe.csv', index=False)
pca_df.to_csv('selected_pca.csv', index=False)
dt_df.to_csv('selected_dt.csv', index=False)
cfs_df.to_csv('selected_cfs.csv', index=False)

print("[SUCCESS] Feature selection completed and saved.")

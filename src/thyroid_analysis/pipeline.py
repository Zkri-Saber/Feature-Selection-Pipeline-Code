
import pandas as pd
from sklearn.model_selection import train_test_split
from thyroid_analysis.data_loader import load_dataset
from thyroid_analysis.preprocessing import apply_knn_imputation, map_diagnostic_group, normalize_features, apply_smote

def load_consensus_features(path='results/features/consensus_features.txt'):
    with open(path, 'r') as f:
        features = f.read().splitlines()
    return features

def run_pipeline(filename='real_dataset_knn_imputed.csv'):
    print("🔄 Running Full Pipeline: Load, Preprocess, Normalize, SMOTE")

    # Load and preprocess dataset
    df = load_dataset(filename)
    df = apply_knn_imputation(df)
    df['Diagnostic Group Code'] = map_diagnostic_group(df['Diagnostic Group Code'])
    df.dropna(subset=['Diagnostic Group Code'], inplace=True)

    # Step 1: Load selected features
    selected_features = load_consensus_features()
    X = df[selected_features]
    y = df['Diagnostic Group Code']

    # Step 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Step 3: Normalize
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    # Step 4: SMOTE
    X_resampled, y_resampled = apply_smote(X_train_scaled, y_train)

    print("✅ Step 1–4 completed: Data ready for modeling.")
    return X_resampled, y_resampled, X_test_scaled, y_test

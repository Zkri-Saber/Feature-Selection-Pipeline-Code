from thyroid_analysis.feature_selection import run_feature_selection
from thyroid_analysis.pipeline import run_pipeline

def main():
    print("=== Starting Thyroid Disease Feature Selection ===")
    
    # Step 1: Run the feature selection pipeline
    run_feature_selection(filename='real_dataset_knn_imputed.csv')

    # Step 2: Run the full pipeline
    run_pipeline()

    # Inform the user of the next step
    print("\n📌 Feature selection is completed.")
    print("👉 To continue, run: python src/train_models.py for model training and evaluation.")
    print("============================================================")

if __name__ == "__main__":
    main()

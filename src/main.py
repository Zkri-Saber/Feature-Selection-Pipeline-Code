from thyroid_analysis.feature_selection import run_feature_selection

def main():
    print("=== Starting Thyroid Disease Feature Selection ===")
    run_feature_selection(filename='real_dataset_knn_imputed.csv')

if __name__ == "__main__":
    main()

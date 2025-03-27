import numpy as np
import pandas as pd

# Hardcoded file names (adjust paths as needed)
predictions_file = "outputs/out_val_v2.csv"
ground_truth_file = "resource/asnlib/publicdata/yelp_val.csv"

DEFAULT_VOTE = 3.5  # Default rating to use for misses if desired

if __name__ == '__main__':
    # Load the CSV files
    predictions_df = pd.read_csv(predictions_file)
    ground_truth_df = pd.read_csv(ground_truth_file)

    # Merge the dataframes on user_id and business_id
    # Use a left join with ground_truth_df as the left dataframe to ensure all ground truth rows remain
    merged_df = pd.merge(ground_truth_df, predictions_df, on=['user_id', 'business_id'], how='left')

    # Count how many predictions are missing
    misses = merged_df['prediction'].isna().sum()
    print(f"Misses (No predictions found): {misses}")

    # Optionally fill missing predictions with a default value so metrics can be computed
    # If you prefer to exclude these from metric calculations, you could instead filter them out.
    merged_df['prediction'].fillna(DEFAULT_VOTE, inplace=True)

    # Calculate absolute differences
    merged_df['abs_diff'] = abs(merged_df['prediction'] - merged_df['stars'])

    # 1. Mean Absolute Error (MAE)
    mae = merged_df['abs_diff'].mean()
    print(f"\nMean Absolute Error (MAE): {mae}")

    print('Following differences are calculated as p - a where p is predicted and a is actual')

    # 2. Count distribution for absolute differences (absolute error bins)
    bins = [0, 1, 2, 3, 4, np.inf]  # Bin edges
    labels = ['>=0 and <1', '>=1 and <2', '>=2 and <3', '>=3 and <4', '>=4']
    merged_df['abs_diff_bin'] = pd.cut(merged_df['abs_diff'], bins=bins, labels=labels, right=False)
    diff_counts_abs = merged_df['abs_diff_bin'].value_counts()

    print("\nDistribution of Absolute Differences:")
    for label, count in diff_counts_abs.items():
        print(f"{label}: n = {count}")

    # Distribution of raw differences (prediction - actual)
    diff = merged_df['prediction'] - merged_df['stars']
    bins = [-4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]  # Bin edges
    labels = ['>=-4 and <-3', '>=-3 and <-2', '>=-2 and <-1', '>=-1 and <0', '>=0 and <1', '>=1 and <2', '>=2 and <3',
              '>=3 and <4', '>=4']
    merged_df['diff_bin'] = pd.cut(diff, bins=bins, labels=labels, right=False)
    diff_counts = merged_df['diff_bin'].value_counts()

    print("\nDistribution of Differences (p - a):")
    for label, count in diff_counts.items():
        print(f"{label}: n = {count}")

    # 3. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(((merged_df['prediction'] - merged_df['stars']) ** 2).mean())
    print(f"\nRoot Mean Squared Error (RMSE): {rmse}")

    # Print Number of misses
    print(f"Number of misses: {misses}")

import sys
import time
import numpy as np
from pyspark import SparkContext
from xgboost import XGBRegressor
from modules.item_based_collaborative import item_based_recommendation
from modules.content_based_recommendation import model_based_recommendation

def competition(folder_path, test_file_name, output_file_name):
    sc = SparkContext('local[*]', 'Competition')
    sc.setLogLevel('ERROR')
    print("Running item based recommendation for test data")
    item_based_rdd_test = (item_based_recommendation(f"{folder_path}/yelp_train.csv", test_file_name, sc)
                           .map(lambda row: ((row[0], row[1]), float(row[2]))))
    print("Running content based recommendation for test data")
    model_based_rdd_test = (model_based_recommendation(folder_path, test_file_name, sc)
                            .map(lambda row: ((row[0], row[1]), float(row[2]))))
    print("Running item based recommendation for validation data")
    item_based_rdd = (item_based_recommendation(f"{folder_path}/yelp_train.csv", f"{folder_path}/yelp_val.csv", sc)
                      .map(lambda row: ((row[0], row[1]), float(row[2]))))
    print("Running content based recommendation for validation data")
    model_based_rdd = (model_based_recommendation(folder_path, f"{folder_path}/yelp_val.csv", sc)
                       .map(lambda row: ((row[0], row[1]), float(row[2]))))
    print("Loading the validation data ground truth")
    validation_rdd = sc.textFile(f"{folder_path}/yelp_val.csv")
    validation_rdd_h = validation_rdd.first()
    validation_rdd = validation_rdd.filter(lambda line: line != validation_rdd_h) \
        .map(lambda line: line.split(',')) \
        .map(lambda row: ((row[0], row[1]), float(row[2])))
    print("Joining the predictions with the ground truth")
    joined_rdd = (item_based_rdd
                  .join(model_based_rdd)
                  .map(lambda row: ((row[0][0], row[0][1]), (row[1][0], row[1][1])))
                  .join(validation_rdd)
                  .map(lambda row: (row[0][0], row[0][1], row[1][0][0], row[1][0][1], row[1][1])))
    print("Converting RDD to NumPy array")
    results = joined_rdd.collect()
    results_array = np.array(results)
    features = results_array[:, 2:4].astype(float)
    labels = results_array[:, 4].astype(float)
    print("Training the XGBRegressor")
    xgb = XGBRegressor(max_depth=3, min_child_weight=3, n_estimators=300, gamma=0.1)
    xgb.fit(features, labels)
    print("Joining the predictions for test data")
    joined_rdd_test = (item_based_rdd_test
                       .join(model_based_rdd_test)
                       .map(lambda row: (row[0][0], row[0][1], row[1][0], row[1][1])))
    print("Converting RDD to NumPy array")
    results_test = joined_rdd_test.collect()
    results_test_array = np.array(results_test)
    test_features = results_test_array[:, 2:4].astype(float)
    print("Predicting the ratings for test data")
    y_pred = xgb.predict(test_features)
    y_pred = np.clip(y_pred, 1, 5)
    print("Writing the results to output file")
    with open(output_file_name, 'w') as f:
        f.write('user_id,business_id,prediction\n')
        for i, row in enumerate(results_test_array):
            f.write(f"{row[0]},{row[1]},{y_pred[i]}\n")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: spark-submit competition.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)
    folder_path, test_file_name, output_file_name = sys.argv[1], sys.argv[2], sys.argv[3]
    start = time.time()
    competition(folder_path, test_file_name, output_file_name)
    print(f"Duration: {time.time() - start:.2f}")

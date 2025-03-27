import json
import numpy as np
from xgboost import XGBRegressor

def transform_row(row):
    user_id, business_id, stars = row['user_id'], row['business_id'], row['stars']
    return user_id, business_id, stars

def process_user_data(row):
    return row['user_id'], (row['review_count'], row['useful'], row['funny'], row['cool'], row['fans'],
                            row['average_stars'], row['compliment_hot'], row['compliment_more'],
                            row['compliment_profile'], row['compliment_cute'], row['compliment_list'],
                            row['compliment_note'], row['compliment_plain'], row['compliment_cool'],
                            row['compliment_funny'], row['compliment_writer'], row['compliment_photos'])

def process_business_data(row):
    return row['business_id'], (row['stars'], row['review_count'], row['is_open'])

def process_tip_data(row):
    return row['business_id'], (1, row['likes'])

def process_photo_data(row):
    if row['label'] == 'food':
        vector = (1, 0, 0, 0, 0, 1)
    elif row['label'] == 'inside':
        vector = (0, 1, 0, 0, 0, 1)
    elif row['label'] == 'outside':
        vector = (0, 0, 1, 0, 0, 1)
    elif row['label'] == 'drink':
        vector = (0, 0, 0, 1, 0, 1)
    else:
        vector = (0, 0, 0, 0, 1, 1)
    return row['business_id'], vector

def process_checkin_data(row):
    count, checkinsum = 0, 0
    for checkin in row['time'].values():
        count += 1
        checkinsum += checkin
    average_checkins = checkinsum / count if count != 0 else 0
    return row['business_id'], (average_checkins, count)

def model_based_recommendation(folder_path, test_file_name, sc):
    ratings = sc.textFile(f'{folder_path}/review_train.json').map(json.loads)
    users = sc.textFile(f'{folder_path}/user.json').map(json.loads).map(process_user_data).cache()
    businesses = sc.textFile(f'{folder_path}/business.json').map(json.loads).map(process_business_data).cache()
    tips = sc.textFile(f'{folder_path}/tip.json').map(json.loads).map(process_tip_data).reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1]))
    photo = sc.textFile(f'{folder_path}/photo.json').map(json.loads).map(process_photo_data).reduceByKey(
        lambda a, b: tuple(map(sum, zip(a, b))))
    checkin = sc.textFile(f'{folder_path}/checkin.json').map(json.loads).map(process_checkin_data)
    businesses = businesses.leftOuterJoin(tips).map(
        lambda row: (row[0], (
            row[1][0][0],
            row[1][0][1],
            row[1][0][2],
            row[1][1][0] if row[1][1] is not None else 0,
            row[1][1][1] if row[1][1] is not None else 0
        ))
    )
    businesses = businesses.leftOuterJoin(photo).map(
        lambda row: (row[0], (
            row[1][0][0],
            row[1][0][1],
            row[1][0][2],
            row[1][0][3],
            row[1][0][4],
            *(row[1][1] if row[1][1] is not None else (0, 0, 0, 0, 0, 0))
        ))
    )
    businesses = businesses.leftOuterJoin(checkin).map(
        lambda row: (row[0], (
            row[1][0][0],
            row[1][0][1],
            row[1][0][2],
            row[1][0][3],
            row[1][0][4],
            row[1][0][5],
            row[1][0][6],
            row[1][0][7],
            row[1][0][8],
            row[1][0][9],
            row[1][0][10],
            row[1][1][0] if row[1][1] is not None else 0,
            row[1][1][1] if row[1][1] is not None else 0
        ))
    )
    ratings_transformed = ratings.map(lambda row: transform_row(row)).map(lambda row: (row[0], row[1:])) \
        .join(users).map(lambda row: (row[1][0][0], (row[1][1], row[1][0][1]))) \
        .join(businesses).map(
        lambda row: (*row[1][0][0], *row[1][1], row[1][0][1]))
    X_train, y_train = np.array(ratings_transformed.map(lambda row: row[:-1]).collect(),
                                dtype=np.float32), np.array(ratings_transformed.map(
        lambda row: row[-1]).collect(), dtype=np.float32)
    xgb = XGBRegressor(max_depth=7, min_child_weight=1, n_estimators=300)
    xgb.fit(X_train, y_train)
    test_data = sc.textFile(test_file_name)
    test_data_h = test_data.first()
    test_data = test_data.filter(lambda line: line != test_data_h).map(lambda row: row.split(','))
    test_data_with_features = test_data.join(users).map(lambda row: (row[1][0], (row[1][1], row[0]))) \
        .join(businesses).map(lambda row: (*row[1][0][0], *row[1][1], row[1][0][1], row[0]))
    X_test = np.array(test_data_with_features.collect())
    y_pred = xgb.predict(X_test[:, :-2].astype(np.float32))
    y_pred = np.clip(y_pred, 1, 5)
    y_pred = np.hstack([X_test, y_pred.reshape(y_pred.shape[0], 1)])[:, -3:]
    return sc.parallelize(y_pred)

import heapq

DEFAULT_VOTE = 3.5
NEIGHBORHOOD_SIZE = 14

def calculate_item_similarity(item_a, item_b):
    a_user_ratings, b_user_ratings = item_a[1], item_b[1]
    a_user_dict = dict(a_user_ratings)
    b_user_dict = dict(b_user_ratings)
    co_rated_users = set(a_user_dict.keys()).intersection(b_user_dict.keys())
    if len(co_rated_users) < 2:
        item_a_ave = sum(a_user_dict.values()) / len(a_user_dict) if a_user_dict else DEFAULT_VOTE
        item_b_ave = sum(b_user_dict.values()) / len(b_user_dict) if b_user_dict else DEFAULT_VOTE
        rating_diff = abs(item_a_ave - item_b_ave)
        return 1 - (rating_diff / 5)
    a_ratings = [a_user_dict[user] for user in co_rated_users]
    b_ratings = [b_user_dict[user] for user in co_rated_users]
    a_avg = sum(a_ratings) / len(a_ratings)
    b_avg = sum(b_ratings) / len(b_ratings)
    numerator = sum((a - a_avg) * (b - b_avg) for a, b in zip(a_ratings, b_ratings))
    denominator = ((sum((a - a_avg) ** 2 for a in a_ratings) ** 0.5) *
                   (sum((b - b_avg) ** 2 for b in b_ratings) ** 0.5))
    return 0 if denominator == 0 else numerator / denominator

def predict_rating(user, business, user_business, business_user, item_similarities):
    active_user_ratings = user_business.get(user, None)
    active_business_ratings = business_user.get(business, None)
    if active_user_ratings is None and active_business_ratings is None:
        return user, business, DEFAULT_VOTE
    if active_user_ratings is None and active_business_ratings is not None:
        return user, business, sum([rating for _, rating in active_business_ratings]) / len(active_business_ratings)
    if active_user_ratings is not None and active_business_ratings is None:
        return user, business, sum([rating for _, rating in active_user_ratings]) / len(active_user_ratings)
    similarities = {}
    for business_b, rating in active_user_ratings:
        if (business, business_b) in item_similarities:
            similarity = item_similarities[(business, business_b)]
        else:
            similarity = calculate_item_similarity((business, active_business_ratings),
                                                   (business_b, business_user[business_b]))
            item_similarities[(business, business_b)] = similarity
        similarities[(business, business_b)] = similarity
    most_similar_neighbours = heapq.nlargest(NEIGHBORHOOD_SIZE, similarities.items(), key=lambda x: x[1])
    numerator = 0
    denominator = 0
    for (_, business_b), similarity in most_similar_neighbours:
        business_b_rating = dict(active_user_ratings).get(business_b)
        if business_b_rating:
            numerator += business_b_rating * similarity
            denominator += abs(similarity)
    return user, business, numerator / denominator if denominator != 0 else DEFAULT_VOTE

def item_based_recommendation(train_file_name: str, test_file_name: str, sc):
    raw_train_file = sc.textFile(train_file_name)
    raw_test_file = sc.textFile(test_file_name)
    raw_train_file_header = raw_train_file.first()
    raw_test_file_header = raw_test_file.first()
    training_data = raw_train_file.filter(lambda line: line != raw_train_file_header).map(
        lambda line: line.split(',')).map(lambda row: (row[0], row[1], float(row[2])))
    test_data = raw_test_file.filter(lambda line: line != raw_test_file_header).map(
        lambda line: line.split(',')).map(lambda row: (row[0], row[1])).cache()
    training_user_business = training_data.map(lambda row: (row[0], (row[1], row[2]))).groupByKey().mapValues(
        lambda values: list(values)).collectAsMap()
    training_business_user = training_data.map(lambda row: (row[1], (row[0], row[2]))).groupByKey().mapValues(
        lambda values: list(values)).collectAsMap()
    training_user_business_bc = sc.broadcast(training_user_business)
    training_business_user_bc = sc.broadcast(training_business_user)
    item_similarities = sc.broadcast({})
    predicted_ratings = test_data.map(
        lambda x: predict_rating(x[0], x[1], training_user_business_bc.value, training_business_user_bc.value,
                                 item_similarities.value))
    return predicted_ratings

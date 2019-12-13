import codecs
import csv

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor

HOLDOUT_RATIO = 0.8

relevant_neighbourhoods = ['Greenwich Village', 'Williamsburg', 'Greenpoint', 'East Village', 'Upper West Side',
                           'Lower East Side', 'Midtown', 'Astoria', 'Cypress Hills', 'Tribeca', 'SoHo',
                           'Flatiron District', 'Theater District', 'NoHo', 'Navy Yard', 'Battery Park City',
                           "Prince's Bay", 'Riverdale', 'West Village', 'Long Island City']


def main():
    dev_raw_data = load_data("NYC_Airbnb/development.csv")
    eval_raw_data = load_data("NYC_Airbnb/evaluation.csv")
    crimes = load_data("datasets/Criminality_rate.csv")
    near_crimes_dev = load_data("datasets/dev_500m.csv")
    near_crimes_eval = load_data("datasets/eval_500m.csv")

    crime_data = {}
    for i in range(len(crimes["neighbourhood_group"])):
        crime_data[crimes["neighbourhood_group"][i]] = [int(crimes["population"][i]), int(crimes["crimes"][i]),
                                                        float(crimes["ratio"][i])]

    near_crimes_dev_data = {}
    near_crimes_eval_data = {}
    for i in range(len(near_crimes_dev["id"])):
        near_crimes_dev_data[near_crimes_dev["id"][i]] = near_crimes_dev["crimes"][i]
    for i in range(len(near_crimes_eval["id"])):
        near_crimes_eval_data[near_crimes_eval["id"][i]] = near_crimes_eval["crimes"][i]

    # analyze crimes distribution to evaluate radius
    # it should be distributed over a wide range of values

    plt.boxplot([int(i) for i in near_crimes_eval_data.values()])
    plt.title("Dev data crimes")
    plt.ylabel("Crimes")
    plt.show()

    #

    '''
    • id: a unique identifier of the listing
    • name                                                                              NOT RELEVANT
    • host_id: a unique identifier of the host                                          NOT RELEVANT
    • host_name                                                                         RELEVANT IF AN HOST HAS MORE THAN ONE AIRBNB, NOT IN THIS CASE
    • neighbourhood_group: neighborhood location in the city                            VERY RELEVANT, BUT CORRELATED TO NEIGHBORHOOD
    • neighbourhood: name of the neighborhood                                           VERY RELEVANT 
    • latitude: coordinate expressed as floating point number                           RELEVANT 
    • longitude: coordinate expressed as floating point number                          RELEVANT
    • room_type                                                                         VERY RELEVANT
    • price: price per night expressed in dollars                                       LABEL
    • minimum_nights: minimum nights requested by the host                              RELEVANT
    • number_of_reviews                                                                 RELEVANT
    • last_review: date of the last review expressed as YYYY-MM-DD                      NOT RELEVANT
    • reviews_per_month: average number of reviews per month                            NOT RELEVANT
    • calculated_host_listings_count: amount of listing of the host                     RELEVANT
    • availability_365: number of days when the listing is available for booking        RELEVANT
    '''

    dimensions, neighbourhood_dict, features_dict = encode_data(dev_raw_data, crime_data, near_crimes_dev_data)
    eval_dimensions, _, _ = encode_data(eval_raw_data, crime_data, near_crimes_eval_data, neighbourhood_dict)

    y = [(float(i)) for i in dev_raw_data["price"]]  # getting price values

    # dimensions, y = remove_outliers(dimensions, y, 0, 99)

    # plt.scatter([float(i) for i in dev_raw_data["reviews_per_month"]], np.log10(y))
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(dimensions, y, train_size=HOLDOUT_RATIO, random_state=42)

    # X_train, y_train = remove_outliers(X_train, y_train, 0.1, 99.9)

    '''
    plt.matshow(train_df.corr())
    plt.colorbar()
    plt.show()
    '''

    reg = regression(X_train, X_test, y_train, y_test)

    # plot_features_importance(reg, features_dict)

    y_pred = reg.predict(eval_dimensions)
    # y_pred = [max(i, 0) for i in y_pred]  # bounding
    plot_prediction(np.array(dev_raw_data["latitude"], dtype=float), np.array(dev_raw_data["longitude"], dtype=float),
                    np.log10(y),
                    np.array(eval_raw_data["latitude"], dtype=float), np.array(eval_raw_data["longitude"], dtype=float),
                    np.log10(y_pred))

    print("Price mean: " + str(np.array(y_pred, dtype=float).mean()))

    dump_to_file(np.array(y_pred, dtype=float), eval_raw_data["id"], "predictions.csv")


def load_data(path):
    data = {}
    with codecs.open(path, "r", 'utf-8') as file:
        header = True
        headers = []
        for row in csv.reader(file):
            if header:
                for field in row:
                    data[field] = []
                headers = row.copy()
                header = False
                continue

            skip = False
            for n, row_value in enumerate(row):  # check 0 prices and skip them
                if headers[n] == "price" and row_value == "0":
                    skip = True

            if skip: continue

            for n, row_value in enumerate(row):  # fill missing values with 0
                if headers[n] == "reviews_per_month" and row_value == '':
                    row_value = "0"

                data[headers[n]].append(row_value)

    return data


def encode_data(raw, crime_data, near_crimes, neighbourhood_one_hot_dict=None):
    data = {}
    data["neighbourhood_group"], ng_one_hot_dict = one_hot_encode_data(raw["neighbourhood_group"])

    # data["neighbourhood"], neighbourhood_one_hot_dict = one_hot_encode_data(raw["neighbourhood"],
    # neighbourhood_one_hot_dict)

    data["neighbourhood"] = custom_one_hot(raw["neighbourhood"], relevant_neighbourhoods)

    encoding_dict = {"Shared room": 0, "Private room": 1, "Entire home/apt": 2}
    data["room_type"] = label_encoding(raw["room_type"], encoding_dict)

    data["latitude"] = [float(i) for i in raw["latitude"]]
    data["longitude"] = [float(i) for i in raw["longitude"]]
    data["number_of_reviews"] = [float(i) for i in raw["number_of_reviews"]]
    data["reviews_per_month"] = [float(i) for i in raw["reviews_per_month"]]

    data["minimum_nights"] = [float(i) for i in raw["minimum_nights"]]
    data["calculated_host_listings_count"] = [float(i) for i in raw["calculated_host_listings_count"]]
    data["availability_365"] = [float(i) for i in raw["availability_365"]]

    dimensions = []

    features_mapping = None
    '''
    # used for features importance plot labels
    features_mapping = neighbourhood_one_hot_dict.__class__(map(reversed, neighbourhood_one_hot_dict.items()))
    for key, val in ng_one_hot_dict.items(): features_mapping[(len(features_mapping.keys()))] = key
    features_mapping[(len(features_mapping.keys()))] = "room_type"
    features_mapping[(len(features_mapping.keys()))] = "minimum_nights"
    features_mapping[(len(features_mapping.keys()))] = "calculated_host_listings_count"
    features_mapping[(len(features_mapping.keys()))] = "number_of_reviews"
    features_mapping[(len(features_mapping.keys()))] = "availability_365"
    features_mapping[(len(features_mapping.keys()))] = "criminality_ratio"
    features_mapping[(len(features_mapping.keys()))] = "crimes_nearby"
    features_mapping[(len(features_mapping.keys()))] = "latitude"
    features_mapping[(len(features_mapping.keys()))] = "longitude"
    '''

    for i in range(len(data["room_type"])):
        row = []

        row.extend(data["neighbourhood"][i])
        row.extend(data["neighbourhood_group"][i])

        row.append(data["room_type"][i])

        row.append(data["minimum_nights"][i])
        row.append(data["calculated_host_listings_count"][i])
        # row.append(data["reviews_per_month"][i])
        row.append(data["number_of_reviews"][i])
        row.append(data["availability_365"][i])

        row.append(crime_data[raw["neighbourhood_group"][i]][2])  # add data about criminality by neighbourhood group

        row.append((int(near_crimes[raw["id"][i]]))) if raw["id"][i] in near_crimes.keys() else row.append(0)

        row.append(data["latitude"][i])
        row.append(data["longitude"][i])
        dimensions.append(row)

    return dimensions, neighbourhood_one_hot_dict, features_mapping


def one_hot_encode_data(data, uniques=None):
    if uniques is None:
        uniques = {}
        i = 0
        for value in data:
            if value not in uniques.keys():
                uniques[value] = i
                i += 1

    one_hots = []
    for values in data:
        code = [0] * len(uniques)
        code[uniques[values]] = 1
        one_hots.append(code)

    return one_hots, uniques


def custom_one_hot(data, relevants):
    # used with a precomputed dictionary of most relevant elements

    dic = {}
    i = 0
    for r in relevants:
        dic[r] = i
        i += 1

    one_hots = []
    for value in data:
        code = [0] * len(dic)
        if value in dic.keys():
            code[dic[value]] = 1
        one_hots.append(code)

    return one_hots


def label_encoding(data, dict=None):
    if dict is None:
        dict = {}
        i = 0
        for value in data:
            if value not in dict.keys():
                dict[value] = i
                i += 1

    encoded = []
    for value in data:
        encoded.append(dict[value])

    return encoded


def regression(X_train, X_test, y_train, y_test, degree=2):
    reg = xgb.XGBRegressor()
    reg.fit(X_train, y_train)

    print_score(reg, X_test, y_test)
    y_pred = [max(i, 0) for i in reg.predict(X_test)]  # bounding

    # eval_r2_mse(y_pred=y_pred, y_test=y_test)
    return reg


# NOT USED
def remove_outliers(data, y, min_percentile, max_percentile):
    m = np.percentile(y, min_percentile)
    M = np.percentile(y, max_percentile)

    y_new = [i for i in y if m < i < M]
    data_new = [data[i] for i in range(len(data)) if m < y[i] < M]

    return data_new, y_new


def print_score(reg, x, y):
    r2 = cross_val_score(reg, x, y, cv=5, scoring='r2')
    print("R2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))
    mse = mean_squared_error(y, reg.predict(x))
    print("MSE: %0.2f" % mse)


def plot_prediction(train_lon, train_lat, y_train, test_lon, test_lat, y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    ax[0].grid()
    ax[0].set_title("Training data")
    ax[0].set_xlabel("Longitude")
    ax[0].set_ylabel("Latitude")
    p1 = ax[0].scatter(train_lat, train_lon, c=y_train, s=2, vmin=np.percentile(y_train, 2.5),
                       vmax=np.percentile(y_train, 97.5))
    plt.colorbar(p1, ax=ax[0])

    ax[1].grid()
    ax[1].set_title("Evaluation data")
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")
    p2 = ax[1].scatter(test_lat, test_lon, c=y_pred, s=2, vmin=np.percentile(y_train, 2.5),
                       vmax=np.percentile(y_train, 97.5))
    plt.colorbar(p2, ax=ax[1])
    plt.show()


def plot_features_importance(reg, features_dict):
    fi = reg.feature_importances_
    sorted = np.argsort(fi)

    fi = [fi[i] for i in sorted if fi[i] > 0]
    sorted = sorted[-len(fi):]

    x = np.arange(0, len(sorted))
    width = 0.8

    fig = plt.figure(figsize=(8, 8))
    plt.barh(x, fi, width, label='features_importances')
    plt.yticks(x, [features_dict[i] for i in sorted])
    plt.legend(loc='best')
    plt.title("Feature Information for XGBoost regressor")
    plt.show()


def eval_r2_mse(y_pred, y_test):
    r2 = r2_score(y_test, y_pred)
    print("R2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: %0.2f" % mse)


def dump_to_file(prediction, ids, filename):
    """Dump the evaluated labels to a CSV file."""

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Id', 'Predicted'])

        for n, label in enumerate(prediction):
            writer.writerow([ids[n], label])


if __name__ == "__main__":
    main()

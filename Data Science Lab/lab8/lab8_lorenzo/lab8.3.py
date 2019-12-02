import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

register_matplotlib_converters()


def main():
    data = load_data("summary_of_weather.csv")
    top10 = top10_complete(data)

    sensor_data = filter_means("22508", data)
    fig = plt.figure(figsize=(15, 7))
    plt.plot(sensor_data[:, 1], sensor_data[:, 0], label="22508")
    plt.grid();
    plt.legend(loc="best")
    plt.xlabel("Year");
    plt.ylabel("Mean Temperature")
    plt.show()

    X_train, X_test, y_train, y_test = get_training_data(sensor_data, 30)

    regressor = polynomial_regression(X_train, X_test, y_train, y_test, 2)

    y_pred = regressor.predict(X_test)
    date_pred = sensor_data[-len(y_pred):, 1]
    plt.plot(date_pred, y_test, label="true value")
    plt.plot(date_pred, y_pred, label="predicted")
    plt.legend(loc="best");
    plt.grid()
    plt.xlabel("Month");
    plt.ylabel("Mean Temperature")

    plt.show()

    # autopredict(regressor, sensor_data[-30:, 0], sensor_data[-30:, 1], 90)
    pass


def load_data(path):
    data = {
        "STA": [],
        "Date": [],
        "MaxTemp": [],
        "MinTemp": [],
        "MeanTemp": []
    }
    with open(path, 'r') as file:
        header = True
        for row in csv.reader(file):
            if header:
                header = False
                continue
            data["STA"].append(row[0])
            data["Date"].append(np.datetime64(datetime.strptime(row[1], "%Y-%m-%d")))
            data["MaxTemp"].append(row[4])
            data["MinTemp"].append(row[5])
            data["MeanTemp"].append(row[6])
    return data


def top10_complete(data):
    # evaluating completeness by number of records
    sta = {}
    for id in data["STA"]:
        if id not in sta.keys():
            sta[id] = 1
        else:
            sta[id] += 1

    top10 = sorted(sta.items(), key=lambda d: d[1])[-10:]
    top10.reverse()

    # plot of mean distributions
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(5, 8))
    for n, (sta, _) in enumerate(top10):
        ax = axes[n % 5][int(n / 5)]
        means = [float(data["MeanTemp"][i]) for i in range(len(data["STA"])) if data["STA"][i] == sta]
        ax.hist(means, label=sta, alpha=1, bins=50)
        ax.set_title(sta)
    plt.show()

    return top10


def filter_means(sta, data):
    return np.array(
        [(float(data["MeanTemp"][i]), data["Date"][i]) for i in range(len(data["MeanTemp"])) if data["STA"][i] == sta])


def get_training_data(data, window):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    train_chunk = [float(i[0]) for i in data if i[1] < np.datetime64("1944-12-31")] # training on data before 1945
    test_chunk = [float(i[0]) for i in data if i[1] >= np.datetime64("1944-12-31")] # evaluationg on 1945 data
    for i in range(len(train_chunk) - window - 1):
        X_train.append(train_chunk[i:i + window])   # building the moving window
        y_train.append(train_chunk[i + window + 1])

    for i in range(len(test_chunk) - window - 1):
        X_test.append(train_chunk[i:i + window])
        y_test.append(train_chunk[i + window + 1])

    # mean = np.mean(data[:, 0])
    # max = np.max(data[:, 0])
    # min = np.min(data[:, 0])

    # normalization
    # X_train = (np.array(X_train) - mean) / ((max - min) / 2)
    # y_train = (np.array(y_train) - mean) / ((max - min) / 2)
    # X_test = (np.array(X_test) - mean) / ((max - min) / 2)
    # y_test = (np.array(y_test) - mean) / ((max - min) / 2)

    return X_train, X_test, y_train, y_test


def polynomial_regression(X_train, X_test, y_train, y_test, degree):
    # Regression (computed with all data)
    reg = make_pipeline(PolynomialFeatures(degree), RandomForestRegressor(n_estimators=10))
    reg.fit(X_train, y_train)

    # y_pred = reg.predict(X_test)

    print_score(reg, X_test, y_test)

    return reg


def print_score(reg, x, y):
    r2 = cross_val_score(reg, x, y, cv=5, scoring='r2')
    print("R2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))
    mse = mean_squared_error(y, reg.predict(x))
    print("MSE: %0.2f" % mse)

# NOT REQUIRED
# predict more than one day using a window based on previous prediction (not suggested in real cases)
def autopredict(reg, last_window, start_window_date, length):
    win_len = len(last_window)  # length of window
    predict_date = start_window_date[-1:][0]  # first date of predictions
    prediction = last_window.copy().tolist()
    predicted_date = []
    for i in range(length):
        prediction.append(reg.predict([prediction[-win_len:], ])[0])
        predicted_date.append(predict_date)
        predict_date += np.timedelta64(1, 'D')

    plt.plot(start_window_date, last_window)
    plt.plot(predicted_date, prediction[win_len:])
    plt.show()


if __name__ == "__main__":
    main()

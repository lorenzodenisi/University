import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def main():
    # plot functions
    x = np.linspace(-20, 20, 100)
    y1 = f1(x)
    y2 = f2(x)
    y3 = f3(x)

    """
    plt.plot(x, y1, label="f1")
    plt.plot(x, y2, label="f2")
    plt.plot(x, y3, label="f3")
    plt.legend(loc="upper_left")

    plt.show()
    """
    X_train, X_test, y_train, y_test = get_function_data(f3, tr=20, n_samples=100)
    # plt.plot(X_test, y_test)
    # plt.show()

    # lr = RandomForestRegressor(n_estimators=10).fit(X_train[:, np.newaxis], y_train[:, np.newaxis])
    l_regressor = LinearRegression()
    rf_regressor = RandomForestRegressor(n_estimators=10)
    lasso_regressor = Lasso(alpha=0.5, tol=0.2)
    ml_regressor = MLPRegressor()
    ridge_regressor = Ridge()

    # y_l = apply_regression(l_regressor, X_train, X_test, y_train, y_test)
    # y_rf = apply_regression(rf_regressor, X_train, X_test, y_train, y_test)
    # y_lasso = apply_regression(lasso_regressor, X_train, X_test, y_train, y_test)
    # y_ml = apply_regression(ml_regressor, X_train, X_test, y_train, y_test)
    # y_ridge = apply_regression(ridge_regressor, X_train, X_test, y_train, y_test)
    y_poly = polynomial_regression(X_train, X_test, y_train, y_test, 2)

    plt.show()
    return


def f1(x_):
    x = np.array(x_)
    y = x * np.sin(x)
    y += 2 * x
    return y


def f2(x_):
    x = np.array(x_)
    y = 10 * np.sin(x) + np.power(x, 2)
    return y


def f3(x_):
    x = np.array(x_)
    y = np.sign(x) * (np.power(x, 2) + 300)
    y += 20 * np.sin(x)

    return y


def get_function_data(function, tr, n_samples):
    X = np.linspace(-tr, tr, n_samples)
    y = function(X)
    y = inject_noise(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=30,
                                                        random_state=42,
                                                        shuffle=True)
    y_test = y_test[X_test.argsort()]
    X_test.sort()

    return X_train, X_test, y_train, y_test


def apply_regression(regressor, X_train, X_test, y_train, y_test):
    regressor.fit(X_train[:, np.newaxis], y_train[:, np.newaxis])
    y_pred = regressor.predict(X_test[:, np.newaxis])
    plt.scatter(X_test, y_test, label="correct")
    plt.plot(X_test, y_pred, label="prediction")
    plt.legend(loc="best")

    print_score(regressor, X_test, y_test)

    return y_pred


def polynomial_regression(X_train, X_test, y_train, y_test, degree):
    # Regression (computed with all data)
    reg = make_pipeline(PolynomialFeatures(degree), RandomForestRegressor())
    reg.fit(X_train[:, np.newaxis], y_train)

    y_pred = reg.predict(X_test[:, np.newaxis])

    plt.scatter(X_test, y_test, label="correct")
    plt.plot(X_test, y_pred, label="prediction")
    plt.legend(loc='best')

    print_score(reg, X_test, y_test)

    return y_pred


def plot_regression_line(reg, label="Regression line", start=0, stop=5):
    # For plots generate 50 linearly spaced samples between start and stop
    x_reg = np.linspace(start, stop, 50)
    y_reg = reg.predict(x_reg[:, np.newaxis])
    plt.plot(x_reg, y_reg, label=label)


def print_score(reg, x, y):
    r2 = cross_val_score(reg, x[:, np.newaxis], y, cv=5, scoring='r2')
    print("R2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))


def inject_noise(y):
    """Add a random noise drawn from a normal distribution."""
    return y + np.random.normal(0, 50, size=y.size)


if __name__ == "__main__":
    main()

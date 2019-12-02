from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns


def main():
    X, y, coeffs = make_regression(n_samples=2000, random_state=42, noise=0.5, n_features=20, n_informative=3,
                                   coef=True)

    # sns.heatmap(pd.DataFrame(X).corr())
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    regressor = polynomial_regression(X_train, X_test, y_train, y_test, 2)
    print("Informative feature coefficients: (ground truth):" + str([str(c) for c in coeffs if c > 0]))
    idx = [i for i, c in enumerate(coeffs) if c > 0]
    print("Coefficients of regressor")
    # Actually the polynomial representation has more than 20 coefficients because it has degree 2
    # so are present all the features squared and the products of each feature with each other feature
    for i in idx:
        input = np.zeros(20)
        input[i] = 1
        print("x" + str(i) + ": " + str(regressor.predict([input, ])))

    # Try with linear regressor
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print_score(lr, X_test, y_test)
    print("Linear regressor coefficients")
    print(str(lr.coef_))


def polynomial_regression(X_train, X_test, y_train, y_test, degree):
    # Regression (computed with all data)
    reg = make_pipeline(PolynomialFeatures(degree), RandomForestRegressor(n_estimators=10))
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    print_score(reg, X_test, y_test)

    return reg


def print_score(reg, x, y):
    r2 = cross_val_score(reg, x, y, cv=5, scoring='r2')
    print("R2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))


if __name__ == "__main__":
    main()

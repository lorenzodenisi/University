import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def main():
    iris = [[], [], [], [], []]
    measurements = ["sepal length", "sepal width", "petal length", "petal width"]
    load_file(iris)
    point2(iris, measurements)
    iris_types = point3(iris, measurements)
    show_plots(iris, measurements, iris_types)


def load_file(iris):
    with open("iris.csv") as f:
        for row in csv.reader(f):
            if len(row) == 5:  # only do this if the number of columns is 5, as expected
                for i in range(4):  # the 4 measurements should be converted to float
                    iris[i].append(float(row[i]))
                # position 4 is the iris type, which is to be kept as a string
                iris[4].append(row[4])


def mean(x):
    return sum(x)/len(x)


def std(x):
    u = mean(x)
    return (mean([(x_i - u) ** 2 for x_i in x]))**0.5


def point2(iris, measurements):
    for i, m in enumerate(measurements):
        print(f"{m} mean: {mean(iris[i]):.4f}, std: {std(iris[i]):.4f}")
    print()


def point3(iris, measurements):
    iris_types = set(iris[4])
    for i, m in enumerate(measurements):
        print(m)
        for iris_type in iris_types:
            # For each measurement and for each iris type, build a list of values
            values = [v for v, t in zip(iris[i], iris[4]) if t == iris_type]
            print(f"{iris_type} {mean(values):.4f} {std(values):.4f}")
        print()
    print()
    return iris_types


def show_plots(iris, measurements, iris_types):
    colors = ['b', 'g', 'r']
    for i, m in enumerate(measurements):
        plt.figure()
        plt.title(m)
        for iris_type, color in zip(iris_types, colors):
            # For each measurement and for each type of iris, build a list of values
            values = [v for v, t in zip(iris[i], iris[4]) if t == iris_type]
            plt.hist(values, density=True, alpha=0.2, color=color)
            u = mean(values)
            s = std(values)
            x = np.linspace(u - 5 * s, u + 5 * s, 100)
            plt.plot(x, norm(u, s).pdf(x), label=iris_type, color=color)
            plt.xlabel(f"{m} (cm)")
            plt.ylabel("density")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
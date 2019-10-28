import csv
import numpy

"""
Each rows is a digit
For each digit there are 785 values
First one is the real digit, the other 784 are the pixels
"""


def load_data(file_name, mnist, labels):
    with open(file_name) as f:
        for cols in csv.reader(f):
            labels.append(int(cols.pop(0)))
            mnist.append(list(map(int, cols)))


def get_char(pixel):
    ranges = {
        (0, 64): " ",
        (64, 128): ".",
        (128, 192): "*",
        (192, 256): "#"
    }
    for (a, b), ch in ranges.items():
        if a <= pixel < b:
            return ch


def print_digit(mnist, digit):
    chars = list(map(get_char, mnist[digit]))
    for i in range(28): # iterate over rows
        for j in range(28): # iterate over columns
            print(chars[i*28+j], end="")
        print()


def euclidean_distance(x, y):
    return sum([ (x_i - y_i) ** 2 for x_i, y_i in zip(x, y) ]) ** 0.5


def main():
    file_name = 'mnist_test.csv'
    mnist = []
    labels = []
    load_data(file_name, mnist, labels)
    print_digit(mnist, 0)


if __name__ == '__main__':
    main()

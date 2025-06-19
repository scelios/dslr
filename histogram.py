import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from useful.math import count_, mean_, std_, min_, max_, percentile_
from useful.csv import load_csv, find_position, putNonNumericalToNaN
def parser():
    parser = argparse.ArgumentParser(
        description="Show a histogram from a CSV file.",
        epilog="Example: python histogram.py datasets/dataset_train.csv",
    )
    parser.add_argument("file", type=str, help="file Path")
    args = parser.parse_args()
    return args

def histogram_plot(X, legend, xLabel, yLabel, title, position):

    h1 = X[:position[0]]
    h1 = h1[~np.isnan(h1)]
    plt.hist(h1, color='red', alpha=0.5, label=legend[0])
    h2 = X[position[0]:position[1]]
    h2 = h2[~np.isnan(h2)]
    plt.hist(h2, color='yellow', alpha=0.5, label=legend[1])
    h3 = X[position[1]:position[2]]
    h3 = h3[~np.isnan(h3)]
    plt.hist(h3, color='blue', alpha=0.5, label=legend[2])
    h4 = X[position[2]:]
    h4 = h4[~np.isnan(h4)]
    plt.hist(h4, color='green', alpha=0.5, label=legend[3])
    plt.legend()
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def histogram(filename):
    dataset = load_csv(filename)
    data = dataset[1:, :]
    data = data[data[:, 1].argsort()]
    legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    position = find_position(data, legend)
    data = putNonNumericalToNaN(data)

    # X = np.array(data[:, 16], dtype=float)
    # histogram_plot(X, legend, xLabel=dataset[0, 16], yLabel='Number of student', title=dataset[0, 16], position=position)
    # return

    for i in range (8, len(data[0])):
        X = np.array(data[:, i], dtype=float)
        histogram_plot(X, legend, xLabel=dataset[0, i], yLabel='Number of student', title=dataset[0, i], position=position)


def main():
    args = parser()
    file = args.file
    if not file:
        print("Please enter the filename")
    elif not file.endswith('.csv'):
        print("Please enter a csv file")
        return
    try:
        histogram(file)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        return
    except Exception as e:
        print(f"Error processing the file: {e}")
        return

if __name__ == '__main__':
    main()
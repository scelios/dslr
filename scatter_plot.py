#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from useful.math import count_, mean_, std_, min_, max_, percentile_
from useful.csv import load_csv, find_position, putNonNumericalToNaN
def parser():
    parser = argparse.ArgumentParser(
        description="Show a scatter plot from a CSV file.",
        epilog="Example: python scatter_plot.py datasets/dataset_train.csv",
    )
    parser.add_argument("file", type=str, help="file Path")
    args = parser.parse_args()
    return args

def scatter_plot(X, y, legend, xlabel, ylabel, position):
  plt.scatter(X[:position[0]], y[:position[0]], color='red', alpha=0.5)
  plt.scatter(X[position[0]:position[1]], y[position[0]:position[1]], color='yellow', alpha=0.5)
  plt.scatter(X[position[1]:position[2]], y[position[1]:position[2]], color='blue', alpha=0.5)
  plt.scatter(X[position[2]:], y[position[2]:], color='green', alpha=0.5)

  plt.legend(legend, loc='upper right', frameon=False)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def scatterPlot(filename):
    dataset = load_csv(filename)
    data = dataset[1:, :]
    data = data[data[:, 1].argsort()]
    legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    position = find_position(data, legend)
    data = putNonNumericalToNaN(data)

    for i in range (7, len(data[0])):
        for j in range(8, len(data[0])):
            if (i == j):
                continue
            X = np.array(data[:, i], dtype=float)
            Y = np.array(data[:, j], dtype=float)
            scatter_plot(X, Y, legend=legend, xlabel=dataset[0, i], ylabel=dataset[0, j], position=position)

def main():
    args = parser()
    file = args.file
    if not file:
        print("Please enter the filename")
    elif not file.endswith('.csv'):
        print("Please enter a csv file")
        return
    try:
        scatterPlot(file)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        return
    except Exception as e:
        print(f"Error processing the file: {e}")
        return

if __name__ == '__main__':
    main()
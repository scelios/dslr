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
        description="Show a pair plot from a CSV file.",
        epilog="Example: python pair_plot.py datasets/dataset_train.csv",
    )
    parser.add_argument("file", type=str, help="file Path")
    parser.add_argument("-o", "--output", type=str, help="Output file path (optional, if not specified will show the plot)", default=None)
    args = parser.parse_args()
    return args

def pair_plot(data, headers, legend, position, output_file=None):
    """
    Create a pair plot (scatter plot matrix) for all numerical features.
    This helps visualize relationships between features and identify which ones
    to use for logistic regression.
    """
    colors = ['red', 'yellow', 'blue', 'green']
    house_colors = []
    for i in range(len(data)):
        if i < position[0]:
            house_colors.append(colors[0])  # Gryffindor
        elif i < position[1]:
            house_colors.append(colors[1])  # Hufflepuff
        elif i < position[2]:
            house_colors.append(colors[2])  # Ravenclaw
        else:
            house_colors.append(colors[3])  # Slytherin
    numerical_features = []
    feature_names = []

    for i in range(6, len(headers)):
        feature_data = np.array(data[:, i], dtype=float)
        if not np.all(np.isnan(feature_data)):
            numerical_features.append(feature_data)
            feature_names.append(headers[i])

    n_features = len(numerical_features)
    fig, axes = plt.subplots(n_features, n_features, figsize=(20, 20))
    fig.suptitle('Pair Plot - Feature Relationships by Hogwarts House', fontsize=16)
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if i == j:
                for k, (house_name, color) in enumerate(zip(legend, colors)):
                    if k == 0:
                        mask = np.arange(len(data)) < position[0]
                    elif k == 1:
                        mask = (np.arange(len(data)) >= position[0]) & (np.arange(len(data)) < position[1])
                    elif k == 2:
                        mask = (np.arange(len(data)) >= position[1]) & (np.arange(len(data)) < position[2])
                    else:
                        mask = np.arange(len(data)) >= position[2]

                    feature_subset = numerical_features[i][mask]
                    feature_subset = feature_subset[~np.isnan(feature_subset)]
                    ax.hist(feature_subset, bins=20, alpha=0.5, color=color, label=house_name)

                if i == 0:
                    ax.legend(loc='upper right', fontsize=6)
            else:
                x_data = numerical_features[j]
                y_data = numerical_features[i]
                for k, (house_name, color) in enumerate(zip(legend, colors)):
                    if k == 0:
                        mask = np.arange(len(data)) < position[0]
                    elif k == 1:
                        mask = (np.arange(len(data)) >= position[0]) & (np.arange(len(data)) < position[1])
                    elif k == 2:
                        mask = (np.arange(len(data)) >= position[1]) & (np.arange(len(data)) < position[2])
                    else:
                        mask = np.arange(len(data)) >= position[2]

                    x_subset = x_data[mask]
                    y_subset = y_data[mask]

                    valid_mask = ~(np.isnan(x_subset) | np.isnan(y_subset))
                    x_subset = x_subset[valid_mask]
                    y_subset = y_subset[valid_mask]

                    ax.scatter(x_subset, y_subset, alpha=0.3, s=1, color=color)

            if j == 0:
                ax.set_ylabel(feature_names[i], fontsize=8)
            else:
                ax.set_yticklabels([])

            if i == n_features - 1:
                ax.set_xlabel(feature_names[j], fontsize=8, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Pair plot saved to {output_file}")
    else:
        plt.show()

def pairPlot(filename, output_file=None):
    dataset = load_csv(filename)
    headers = dataset[0, :]
    data = dataset[1:, :]
    data = data[data[:, 1].argsort()]
    legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    position = find_position(data, legend)
    data = putNonNumericalToNaN(data)
    pair_plot(data, headers, legend, position, output_file)

def main():
    args = parser()
    file = args.file
    if not file:
        print("Please enter the filename")
    elif not file.endswith('.csv'):
        print("Please enter a csv file")
        return
    try:
        pairPlot(file, args.output)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        return
    except Exception as e:
        print(f"Error processing the file: {e}")
        return

if __name__ == '__main__':
    main()

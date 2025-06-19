import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from useful.math import count_, mean_, std_, min_, max_, percentile_
from useful.csv import load_csv, putNonNumericalToNaN
def parser():
    parser = argparse.ArgumentParser(
        description="Describe a dataset from a CSV file.",
        epilog="Example: python describe.py datasets/dataset_train.csv",
    )
    parser.add_argument("file", type=str, help="file Path")
    args = parser.parse_args()
    return args



def describe_dataset(filename):
    dataset = load_csv(filename)

    dataset = putNonNumericalToNaN(dataset)

    
    features = dataset[0]
    dataset = dataset[1:, :]
    print(f'{"":15} |{"Count":>13} |{"Mean":>13} |{"Std":>13} |{"Min":>13} |{"25%":>13} |{"50%":>13} |{"75%":>13} |{"Max":>13}')
    for i in range(0, len(features)):
        print(f'{features[i]:15.15}', end=' |')
        try:
            data = np.array(dataset[:, i], dtype=float)
            data = data[~np.isnan(data)]
            if not data.any():
                raise Exception()
            print(f'{count_(data):>13f}', end=' |')
            print(f'{mean_(data):>13f}', end=' |')
            print(f'{std_(data):>13f}', end=' |')
            print(f'{min_(data):>13f}', end=' |')
            print(f'{percentile_(data, 25):>13f}', end=' |')
            print(f'{percentile_(data, 50):>13f}', end=' |')
            print(f'{percentile_(data, 75):>13f}', end=' |')
            print(f'{max_(data):>13f}')
        except:
            print(f'{count_(dataset[:, i]):>13f}', end=' |')
            print(f'{"No numerical value to display":>60}')


def main():
    args = parser()
    file = args.file
    if not file:
        print("Please enter the filename")
    elif not file.endswith('.csv'):
        print("Please enter a csv file")
        return
    # df = load_csv(file)
    # if df is not np:
    try:
        describe_dataset(file)
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        return
    except Exception as e:
        print(f"Error processing the file: {e}")
        return

if __name__ == '__main__':
    main()
import numpy as np
import csv

def load_csv(filename = './datasets/dataset_train.csv'):
    dataset = list()
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        try:
            # if not any(reader):
            #     raise ValueError(f"File {filename} is empty or not a valid CSV file.")
            for _ in reader:
                row = list()
                for value in _:
                    # try:
                    #     value = float(value)
                    # except ValueError: 
                    #     # set value to NaN if it is not on first row 
                    #     if reader.line_num > 1:
                    #         value = np.nan
                    row.append(value)
                dataset.append(row)
        except csv.Error as e:
            print(f'file {filename}, line {reader.line_num}: {e}')
    return np.array(dataset, dtype=object)

def find_position(data, legend):
    position = [0, 0, 0]
    index = 0
    # Find the position of each house in the data
    for i in range(len(data)):
        if data[i][1] == legend[index]:
            continue
        else:
            position[index] = i
            index += 1
            if index >= len(legend):
                break
    return position

def putNonNumericalToNaN(dataset):
    # set non-numerical values to NaN
    for i in range(1, len(dataset)):
        for j in range(len(dataset[i])):
            try:
                dataset[i][j] = float(dataset[i][j])
            except ValueError:
                dataset[i][j] = np.nan
    return dataset
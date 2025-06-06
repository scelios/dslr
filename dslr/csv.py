import numpy as np
import csv

def load_csv(filename):
    dataset = list()
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        try:
            # if not any(reader):
            #     raise ValueError(f"File {filename} is empty or not a valid CSV file.")
            for _ in reader:
                row = list()
                for value in _:
                    try:
                        value = float(value)
                    except ValueError: 
                        # set value to NaN if it is not on first row 
                        if reader.line_num > 1:
                            value = np.nan
                    row.append(value)
                dataset.append(row)
        except csv.Error as e:
            print(f'file {filename}, line {reader.line_num}: {e}')
    return np.array(dataset, dtype=object)
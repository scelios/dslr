import numpy as np

def count_(X):
    """
    Count the number of rows in the DataFrame.
    """
    try:
        X = X.astype('float')
        X = X[~np.isnan(X)]
        return len(X)
    except Exception as e:
        raise ValueError(f"Error counting rows: {e}")

def mean_(X):
    """
    Calculate the mean of each numeric column in the DataFrame.
    """
    total = 0
    count = 0
    try:
        X = X.astype('float')
        for value in X:
            if not np.isnan(value):
                total += value
                count += 1
        if count == 0:
            raise ValueError("No valid numeric values to calculate mean.")
        return total / count
    except Exception as e:
        raise ValueError(f"Error calculating mean: {e}")

def std_(X):
    """
    Calculate the standard deviation of each numeric column in the DataFrame.
    """
    try:
        X = X.astype('float')
        mean_value = mean_(X)
        variance = 0
        count = 0
        for value in X:
            if not np.isnan(value):
                variance += (value - mean_value) ** 2
                count += 1
        if count < 2:
            raise ValueError("Not enough valid numeric values to calculate standard deviation.")
        return np.sqrt(variance / (count - 1))
    except Exception as e:
        raise ValueError(f"Error calculating standard deviation: {e}")

def min_(X):
    """
    Calculate the minimum value of each numeric column in the DataFrame.
    """
    try:
        X = X.astype('float')
        min_value = np.inf
        for value in X:
            if not np.isnan(value) and value < min_value:
                min_value = value
        if min_value == np.inf:
            raise ValueError("No valid numeric values to calculate minimum.")
        return min_value
    except Exception as e:
        raise ValueError(f"Error calculating minimum: {e}")

def max_(X):
    """
    Calculate the maximum value of each numeric column in the DataFrame.
    """
    try:
        X = X.astype('float')
        max_value = -np.inf
        for value in X:
            if not np.isnan(value) and value > max_value:
                max_value = value
        if max_value == -np.inf:
            raise ValueError("No valid numeric values to calculate maximum.")
        return max_value
    except Exception as e:
        raise ValueError(f"Error calculating maximum: {e}")

def percentile_(X, q):
    """
    Calculate the q-th percentile_ of each numeric column in the DataFrame.
    """
    try:
        X = X.astype('float')
        if not (0 <= q <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        sorted_X = np.sort(X[~np.isnan(X)])
        if len(sorted_X) == 0:
            raise ValueError("No valid numeric values to calculate percentile.")
        index = int(np.ceil(q / 100 * len(sorted_X))) - 1
        return sorted_X[index]
    except Exception as e:
        raise ValueError(f"Error calculating percentile: {e}")
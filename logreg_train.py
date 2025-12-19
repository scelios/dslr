#!/usr/bin/env python3
import csv
import numpy as np
import argparse
import json
from useful.csv import load_csv, putNonNumericalToNaN

def parser():
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model for Hogwarts house classification.",
        epilog="Example: python logreg_train.py datasets/dataset_train.csv",
    )
    parser.add_argument("file", type=str, help="Training dataset path")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("-i", "--iterations", type=int, default=1000, help="Number of iterations (default: 1000)")
    parser.add_argument("-o", "--output", type=str, default="weights.json", help="Output weights file (default: weights.json)")
    args = parser.parse_args()
    return args

def sigmoid(z):
    """
    Sigmoid activation function.
    Returns values between 0 and 1.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def hypothesis(X, theta):
    """
    Hypothesis function: h(x) = sigmoid(X * theta)
    """
    return sigmoid(np.dot(X, theta))

def cost_function(X, y, theta):
    """
    Logistic regression cost function (log loss).
    J(θ) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
    """
    m = len(y)
    h = hypothesis(X, theta)
    epsilon = 1e-7
    h = np.clip(h, epsilon, 1 - epsilon)

    cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Gradient descent algorithm to minimize the cost function.
    Updates theta iteratively.
    """
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = hypothesis(X, theta)
        gradient = 1/m * np.dot(X.T, (h - y))
        theta = theta - learning_rate * gradient

        cost = cost_function(X, y, theta)
        cost_history.append(cost)
        if (i + 1) % 100 == 0:
            print(f"  Iteration {i + 1}/{iterations}, Cost: {cost:.6f}")

    return theta, cost_history

def normalize_features(X):
    """
    Normalize features using standardization (z-score normalization).
    Returns normalized features and the parameters (mean, std) for later use.
    """
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1

    X_normalized = (X - mean) / std

    return X_normalized, mean, std

def prepare_data(filename):
    """
    Load and prepare the training data.
    Returns X (features), y (labels), feature_names, and normalization parameters.
    """
    print("Loading dataset...")
    dataset = load_csv(filename)
    headers = dataset[0, :]
    data = dataset[1:, :]
    houses = []
    for row in data:
        houses.append(row[1])  # Hogwarts House is in column 1

    houses = np.array(houses)
    data = putNonNumericalToNaN(data)
    feature_indices = []
    feature_names = []

    for i in range(6, len(headers)):
        feature_data = np.array(data[:, i], dtype=float)
        if np.sum(~np.isnan(feature_data)) > len(feature_data) * 0.5:
            feature_indices.append(i)
            feature_names.append(headers[i])

    print(f"Selected features: {feature_names}")
    X = np.zeros((len(data), len(feature_indices)))
    for i, idx in enumerate(feature_indices):
        X[:, i] = np.array(data[:, idx], dtype=float)
    for i in range(X.shape[1]):
        col_mean = np.nanmean(X[:, i])
        X[np.isnan(X[:, i]), i] = col_mean
    print("Normalizing features...")
    X_normalized, mean, std = normalize_features(X)
    X_normalized = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

    return X_normalized, houses, feature_names, mean, std

def train_one_vs_all(X, y, houses, learning_rate, iterations):
    """
    Train one-vs-all logistic regression for multi-class classification.
    Returns a dictionary of weights for each house.
    """
    unique_houses = np.unique(houses)
    all_theta = {}

    print(f"\nTraining one-vs-all models for {len(unique_houses)} houses...")

    for house in unique_houses:
        print(f"\nTraining model for {house}:")
        y_binary = (y == house).astype(float)
        theta = np.zeros(X.shape[1])
        theta, cost_history = gradient_descent(X, y_binary, theta, learning_rate, iterations)
        all_theta[house] = theta.tolist()
        print(f"  Final cost: {cost_history[-1]:.6f}")

    return all_theta

def save_weights(weights, feature_names, mean, std, output_file):
    """
    Save trained weights and normalization parameters to a JSON file.
    """
    model = {
        'weights': weights,
        'feature_names': feature_names,
        'normalization': {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(model, f, indent=2)

    print(f"\nModel saved to {output_file}")

def train(filename, learning_rate, iterations, output_file):
    """
    Main training function.
    """
    X, y, feature_names, mean, std = prepare_data(filename)

    print(f"\nDataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features (+ bias): {X.shape[1]}")
    weights = train_one_vs_all(X, y, y, learning_rate, iterations)
    save_weights(weights, feature_names, mean, std, output_file)
    print("\nTraining complete!")

def main():
    args = parser()
    file = args.file

    if not file:
        print("Please enter the filename")
        return
    elif not file.endswith('.csv'):
        print("Please enter a csv file")
        return
    if args.iterations <= 0:
        print(f"Error: iterations must be positive (you provided: {args.iterations})")
        print("Usage: python3 logreg_train.py datasets/dataset_train.csv -i 1000")
        return
    if args.learning_rate <= 0:
        print(f"Error: learning_rate must be positive (you provided: {args.learning_rate})")
        return
    try:
        train(file, args.learning_rate, args.iterations, args.output)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()

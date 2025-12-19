#!/usr/bin/env python3
import csv
import numpy as np
import argparse
import json
from useful.csv import load_csv, putNonNumericalToNaN

def parser():
    parser = argparse.ArgumentParser(
        description="Predict Hogwarts houses using trained logistic regression model.",
        epilog="Example: python logreg_predict.py datasets/dataset_test.csv weights.json",
    )
    parser.add_argument("dataset", type=str, help="Test dataset path")
    parser.add_argument("weights", type=str, help="Trained weights file (JSON)")
    parser.add_argument("-o", "--output", type=str, default="houses.csv", help="Output predictions file (default: houses.csv)")
    args = parser.parse_args()
    return args

def sigmoid(z):
    """
    Sigmoid activation function.
    """
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def load_model(weights_file):
    """
    Load trained model from JSON file.
    """
    with open(weights_file, 'r') as f:
        model = json.load(f)
    return model

def prepare_test_data(filename, feature_names, mean, std):
    """
    Load and prepare test data using the same preprocessing as training.
    """
    print("Loading test dataset...")
    dataset = load_csv(filename)
    headers = dataset[0, :]
    data = dataset[1:, :]
    indices = data[:, 0]
    data = putNonNumericalToNaN(data)
    feature_indices = []
    for feature_name in feature_names:
        try:
            idx = list(headers).index(feature_name)
            feature_indices.append(idx)
        except ValueError:
            print(f"Warning: Feature '{feature_name}' not found in test dataset")
            return None
    X = np.zeros((len(data), len(feature_indices)))
    for i, idx in enumerate(feature_indices):
        X[:, i] = np.array(data[:, idx], dtype=float)
    for i in range(X.shape[1]):
        X[np.isnan(X[:, i]), i] = mean[i]
    X_normalized = (X - mean) / std
    X_normalized = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

    return X_normalized, indices

def predict(X, weights):
    """
    Make predictions using trained weights.
    Returns predicted house for each sample.
    """
    houses = list(weights.keys())
    predictions = []
    probabilities = np.zeros((X.shape[0], len(houses)))
    for i, house in enumerate(houses):
        theta = np.array(weights[house])
        probabilities[:, i] = sigmoid(np.dot(X, theta))
    for i in range(X.shape[0]):
        max_prob_idx = np.argmax(probabilities[i])
        predictions.append(houses[max_prob_idx])
    return predictions

def save_predictions(indices, predictions, output_file):
    """
    Save predictions to CSV file in the required format.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Hogwarts House'])

        for idx, house in zip(indices, predictions):
            try:
                idx = int(float(idx))
            except (ValueError, TypeError):
                pass
            writer.writerow([idx, house])

    print(f"Predictions saved to {output_file}")

def predict_houses(dataset_file, weights_file, output_file):
    """
    Main prediction function.
    """
    # Load trained model
    print("Loading trained model...")
    model = load_model(weights_file)
    weights = model['weights']
    feature_names = model['feature_names']
    mean = np.array(model['normalization']['mean'])
    std = np.array(model['normalization']['std'])
    print(f"Model loaded: {len(weights)} houses, {len(feature_names)} features")
    X, indices = prepare_test_data(dataset_file, feature_names, mean, std)
    if X is None:
        print("Error: Could not prepare test data")
        return
    print(f"Test dataset shape: {X.shape}")
    print("Making predictions...")
    predictions = predict(X, weights)
    save_predictions(indices, predictions, output_file)
    print(f"\nPrediction complete! Predicted {len(predictions)} samples.")

def main():
    args = parser()
    dataset = args.dataset
    weights = args.weights

    if not dataset:
        print("Please enter the dataset filename")
        return
    elif not dataset.endswith('.csv'):
        print("Please enter a csv file for dataset")
        return

    if not weights:
        print("Please enter the weights filename")
        return

    try:
        predict_houses(dataset, weights, args.output)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()

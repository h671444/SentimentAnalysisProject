import os
import pickle
from typing import Tuple, List, Optional, Union
import argparse
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define base paths relative to this script
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
MODELS_BASE_DIR = SCRIPT_DIR / "../models"
PROCESSED_DATA_DIR = SCRIPT_DIR / "../data/processed"
PROCESSED_FILENAME = "amazon_reviews_processed.csv"

def load_artifacts(model_path: str, tokenizer_path: str) -> Tuple[Sequential, Tokenizer]:
    """Loads a saved Keras model and its corresponding tokenizer."""
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    print("Artifacts loaded.")
    return model, tokenizer

def preprocess_for_predict(text_data: Union[str, Series], tokenizer: Tokenizer, maxlen: int) -> np.ndarray:
    """Preprocesses text data using a loaded tokenizer and maxlen."""
    if isinstance(text_data, str):
        text_data = [text_data]

    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return padded_sequences

def predict_sentiment(text: str, model: Sequential, tokenizer: Tokenizer, maxlen: int) -> Tuple[str, float]:
    """Predicts sentiment for a single string of text."""
    processed_text = preprocess_for_predict(text, tokenizer, maxlen)

    prediction = model.predict(processed_text, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    
    return sentiment, confidence

def evaluate_model(model: Sequential, X_test_pad: np.ndarray, y_test: Series,
                   batch_size: int = 64) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates the model on the test set."""
    print(f"\nEvaluating model on {len(X_test_pad)} test samples...")
    test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=1, batch_size=batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Generate predictions
    y_pred_probs = model.predict(X_test_pad, verbose=0)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)

    return test_loss, test_acc, y_pred_classes, y_pred_probs

def generate_text_report(y_test: Series, y_pred_classes: np.ndarray, y_pred_probs: np.ndarray) -> None:
    """Generates and prints a text report of model performance."""
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    # Calculate and print AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC: {roc_auc:.4f}")

def plot_confusion_matrix(y_test: Series, y_pred_classes: np.ndarray) -> None:
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_test: Series, y_pred_probs: np.ndarray) -> None:
    """Plots the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment analysis model.")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "rnn_gru", "rnn_lstm"],
                        help="Type of model to evaluate (cnn, rnn_gru, rnn_lstm)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data used for validation/test split (must match training)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")

    args = parser.parse_args()

    # --- Determine Model Directory --- 
    model_dir_name = args.model
    if args.model == "rnn_gru":
        model_dir_name = "rnn-gru"
    elif args.model == "rnn_lstm":
        model_dir_name = "rnn-lstm"
    model_dir = MODELS_BASE_DIR / model_dir_name
    model_path = str(model_dir / "best_model.keras")
    tokenizer_path = str(model_dir / "tokenizer.pickle")

    # --- Load Artifacts --- 
    try:
        model, tokenizer = load_artifacts(model_path, tokenizer_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Ensure the model and tokenizer exist in the correct directory after training.")
        exit(1)

    # --- Infer Maxlen --- 
    try:
        maxlen = model.input_shape[1]
        if maxlen is None:
             raise ValueError("Could not infer maxlen from model input shape.")
        print(f"Inferred maxlen from model: {maxlen}")
    except Exception as e:
        print(f"ERROR: Could not determine maxlen from model: {e}")
        exit(1)

    # --- Load and Split Data --- 
    data_path = PROCESSED_DATA_DIR / PROCESSED_FILENAME
    if not data_path.exists():
        print(f"ERROR: Processed data file not found at {data_path}")
        exit(1)

    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    df.dropna(subset=['cleaned_review', 'polarity'], inplace=True)

    print(f"Recreating train/test split (test_size={args.val_split}, random_state=42)...")
    X = df['cleaned_review']
    y = df['polarity']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    print(f"Test set size: {len(X_test)}")

    # --- Preprocess Test Data --- 
    print("Preprocessing test data...")
    X_test_pad = preprocess_for_predict(X_test, tokenizer, maxlen)

    # --- Evaluate Model --- 
    test_loss, test_acc, y_pred_classes, y_pred_probs = evaluate_model(
        model, X_test_pad, y_test, batch_size=args.batch_size
    )

    # --- Generate Reports and Plots --- 
    generate_text_report(y_test, y_pred_classes, y_pred_probs)
    plot_confusion_matrix(y_test, y_pred_classes)
    plot_roc_curve(y_test, y_pred_probs)

    print("\nEvaluation complete.")


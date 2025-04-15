import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import numpy as np
from typing import Tuple, List, Optional
from pandas import Series

def load_artifacts(model_path: str, tokenizer_path: str) -> Tuple[Sequential, Tokenizer]:
    # Loads a saved Keras model and its corresponding tokenizer
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Artifacts loaded.")
    return model, tokenizer

def preprocess_for_predict(text_data: Series, tokenizer: Tokenizer, maxlen: int) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(text_data)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return padded_sequences

def evaluate_model(model: Sequential, X_test_pad: np.ndarray, y_test: Series,
                   batch_size: int = 64) -> Tuple[float, float, np.ndarray, np.ndarray]:
    
    print(f"Evaluating model on {len(X_test_pad)} test samples...")
    test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=1, batch_size=batch_size)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    print("Generating predictions...")
    y_pred_probs = model.predict(X_test_pad, batch_size=batch_size, verbose=1)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)

    return test_loss, test_acc, y_pred_classes, y_pred_probs.ravel()

def generate_text_report(y_test: Series, y_pred: np.ndarray, y_pred_probs: Optional[np.ndarray] = None):
    # calculates and prints classification report, confusion matrix, and AUC (if probs provided)
    print("\n--- Evaluation Report --- ")

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ROC AUC (optional)
    if y_pred_probs is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")

    print("--- End Report --- \n")

def plot_confusion_matrix(y_test: Series, y_pred: np.ndarray, class_names: Optional[List[str]] = None):
    # plots the confusion matrix as a heatmap
    if class_names is None:
        class_names = ['Negative', 'Positive']

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test: Series, y_pred_probs: np.ndarray):
    # plots the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


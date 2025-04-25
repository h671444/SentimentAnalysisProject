from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from pandas import Series
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import argparse
import pathlib

# Import model builders
from models.cnn import build_cnn_model
from models.rnn_gru import build_gru_model
from models.rnn_lstm import build_lstm_model

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROCESSED_DATA_DIR = SCRIPT_DIR / "../data/processed"
MODELS_BASE_DIR = SCRIPT_DIR / "../models"
PROCESSED_FILENAME = "amazon_reviews_processed.csv"

def train_model(X_train: Series, y_train: Series, X_val: Series, y_val: Series,
                model_builder: callable,
                model_params: dict,
                tokenizer_params: dict,
                maxlen: int,
                epochs: int = 20,
                batch_size: int = 64,
                learning_rate: float = 0.001,
                model_save_dir: str = '../models/cnn',
                model_name: str = 'best_model') -> Tuple[Sequential, Dict, Tokenizer]:
    """Trains a Keras model with the given data and parameters."""

    print("Tokenizing data...")
    tokenizer = Tokenizer(**tokenizer_params)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    print(f"Padding sequences to maxlen={maxlen}...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen, padding='post')

    print("Building model...")
    model = model_builder(**model_params)

    model.summary() # print model summary to console

    print("Compiling model...")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Define save paths based on model_name and ensure directory exists
    # Note: model_save_dir is now an absolute pathlib.Path object
    run_save_dir = model_save_dir / model_name
    run_save_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = run_save_dir / f'{model_name}.keras'
    tokenizer_save_path = run_save_dir / 'tokenizer.pickle'
    print(f"Model artifacts will be saved to: {run_save_dir}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
        # Save checkpoint as string path
        ModelCheckpoint(filepath=str(model_checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print(f"Starting training (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val_pad, y_val),
                        callbacks=callbacks,
                        verbose=1)

    # save Tokenizer after training completes
    print(f"Saving tokenizer to {tokenizer_save_path}...")
    try:
        with open(tokenizer_save_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"ERROR: Failed to save tokenizer: {e}")

    print("Training complete.")
    return model, history.history, tokenizer

def plot_training_history(history: Dict):
    """Plots training and validation accuracy and loss from a history dictionary."""
    required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    if not all(key in history for key in required_keys):
        print(f"Warning: History dictionary missing required keys: {required_keys}. Cannot plot.")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "rnn-gru", "rnn-lstm"],
                        help="Type of model to train (cnn, rnn-gru, rnn-lstm)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--maxlen", type=int, default=70, help="Maximum sequence length for padding")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Maximum vocabulary size for tokenizer")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension for embedding layer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for model layers")
    parser.add_argument("--model_name", type=str, default="best_model", help="Name for saving model artifacts")

    args = parser.parse_args()

    # --- 1. Load Processed Data ---
    data_path = PROCESSED_DATA_DIR / PROCESSED_FILENAME
    if not data_path.exists():
        print(f"ERROR: Processed data file not found at {data_path}")
        print("Please run src/preprocessing.py first.")
        exit(1)
    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    df.dropna(subset=['cleaned_review', 'polarity'], inplace=True) # Ensure no NAs

    # --- 2. Split Data (Train/Validation) ---
    print(f"Splitting data into training and validation sets (validation size: {args.val_split:.1f})...")
    X = df['cleaned_review']
    y = df['polarity']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}")

    # --- 3. Set Up Model Specifics ---
    model_builder = None
    # Base params, vocab size added conditionally
    model_params = {
        'embedding_dim': args.embedding_dim,
        'dropout_rate': args.dropout_rate,
        'maxlen': args.maxlen
    }
    model_save_dir = MODELS_BASE_DIR / args.model

    if args.model == "cnn":
        model_builder = build_cnn_model
        # CNN expects 'max_vocab'
        model_params['max_vocab'] = args.vocab_size
    elif args.model == "rnn-gru":
        model_builder = build_gru_model
        # GRU expects 'max_vocab' (based on its signature)
        model_params['max_vocab'] = args.vocab_size
        # model_params['gru_units'] = 128 # Example
    elif args.model == "rnn-lstm":
        model_builder = build_lstm_model
        # LSTM expects 'vocab_size'
        model_params['vocab_size'] = args.vocab_size
        # model_params['lstm_units'] = 128 # Example
    else:
        print(f"ERROR: Unknown model type '{args.model}'")
        exit(1)

    # Tokenizer always uses the vocab_size argument
    tokenizer_params = {'num_words': args.vocab_size, 'oov_token': '<OOV>'}

    # --- 4. Train the Model ---
    print(f"\n--- Starting Training for {args.model.upper()} Model ---")
    model, history, tokenizer = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_builder=model_builder,
        model_params=model_params,
        tokenizer_params=tokenizer_params,
        maxlen=args.maxlen,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_save_dir=model_save_dir,
        model_name=args.model_name
    )

    # --- 5. Plot Training History ---
    print("\nPlotting training history...")
    plot_training_history(history)

    print(f"\n--- Training for {args.model.upper()} Model Finished Successfully! ---")


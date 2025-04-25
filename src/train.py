from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
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

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROCESSED_DATA_DIR = SCRIPT_DIR / "../data/processed"
MODELS_BASE_DIR = SCRIPT_DIR / "../models"
PROCESSED_FILENAME = "amazon_reviews_processed.csv"

def train_model(X_train: Series, y_train: Series, X_val: Series, y_val: Series,
                model_builder: callable,
                model_params: dict,
                tokenizer_params: dict,
                maxlen: int,
                epochs: int = 10, # Default from notebooks
                batch_size: int = 64,
                learning_rate: float = 0.001,
                model_save_dir: pathlib.Path = None,
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
    model.summary()

    print("Compiling model...")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = model_save_dir / f'{model_name}.keras'
    tokenizer_save_path = model_save_dir / 'tokenizer.pickle'
    print(f"Model artifacts will be saved to: {model_save_dir}")
    print(f"  Model file: {model_checkpoint_path.name}")
    print(f"  Tokenizer file: {tokenizer_save_path.name}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=str(model_checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print(f"Starting training (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val_pad, y_val),
                        callbacks=callbacks,
                        verbose=1)

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
    # Command-line arguments
    parser.add_argument("--model", type=str, required=True, choices=["cnn", "rnn_gru", "rnn_lstm"],
                        help="Type of model to train (cnn, rnn-gru, rnn-lstm)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--maxlen", type=int, default=70, help="Maximum sequence length for padding")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Maximum vocabulary size for tokenizer")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension for embedding layer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for model layers")
    parser.add_argument("--model_name", type=str, default="best_model", help="Name for saving model artifacts")
    # Model-specific structural arguments
    parser.add_argument("--gru_units", type=int, default=64, help="Number of units for GRU layers")
    parser.add_argument("--lstm_units", type=int, default=128, help="Number of units for LSTM layers")
    parser.add_argument("--cnn_kernel_size", type=int, default=5, help="Kernel size for CNN layers")

    args = parser.parse_args()

    # Determine target hyperparameters based on selected model (using table values)
    # Get argparse defaults for comparison later
    default_vocab_size = parser.get_default("vocab_size")
    default_embedding_dim = parser.get_default("embedding_dim")
    default_maxlen = parser.get_default("maxlen")
    default_dropout_rate = parser.get_default("dropout_rate")
    default_lr = parser.get_default("lr")
    default_gru_units = parser.get_default("gru_units")
    default_lstm_units = parser.get_default("lstm_units")
    default_cnn_kernel_size = parser.get_default("cnn_kernel_size")

    # Set model-specific targets from the hyperparameter table
    if args.model == "cnn":
        target_vocab_size = 10000
        target_embedding_dim = 100
        target_maxlen = 70
        target_dropout_rate = 0.3
        target_lr = 0.001
        # Use arg for kernel size if provided, else notebook default
        cnn_filters = [64, 128, 128]
        cnn_kernel_size = args.cnn_kernel_size if args.cnn_kernel_size != default_cnn_kernel_size else 5
    elif args.model == "rnn_gru":
        target_vocab_size = 10000
        target_embedding_dim = 64
        target_maxlen = 70
        target_dropout_rate = 0.3
        target_lr = 0.001
        # Use arg for units if provided, else table default
        gru_units = args.gru_units if args.gru_units != default_gru_units else 64
    elif args.model == "rnn_lstm":
        target_vocab_size = 12000
        target_embedding_dim = 128
        target_maxlen = 80
        target_dropout_rate = 0.4
        target_lr = 0.0007
        # Use arg for units if provided, else table default
        lstm_units = args.lstm_units if args.lstm_units != default_lstm_units else 128
    else:
        # Fallback (should not happen with choices=[...])
        print(f"Warning: Unknown model type '{args.model}', using base defaults.")
        target_vocab_size = default_vocab_size
        target_embedding_dim = default_embedding_dim
        target_maxlen = default_maxlen
        target_dropout_rate = default_dropout_rate
        target_lr = default_lr
        cnn_kernel_size = default_cnn_kernel_size
        gru_units = default_gru_units
        lstm_units = default_lstm_units

    # Determine effective hyperparameters: use user's value IF different from default, else use model-specific target.
    effective_vocab_size = args.vocab_size if args.vocab_size != default_vocab_size else target_vocab_size
    effective_embedding_dim = args.embedding_dim if args.embedding_dim != default_embedding_dim else target_embedding_dim
    effective_maxlen = args.maxlen if args.maxlen != default_maxlen else target_maxlen
    effective_dropout_rate = args.dropout_rate if args.dropout_rate != default_dropout_rate else target_dropout_rate
    effective_lr = args.lr if args.lr != default_lr else target_lr
    # Structural params (gru_units, lstm_units, cnn_kernel_size, cnn_filters) are already determined above

    # Load Data
    data_path = PROCESSED_DATA_DIR / PROCESSED_FILENAME
    if not data_path.exists():
        print(f"ERROR: Processed data file not found at {data_path}")
        print("Please run src/preprocessing.py first.")
        exit(1)
    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    df.dropna(subset=['cleaned_review', 'polarity'], inplace=True)

    # Split Data
    print(f"Splitting data into training and validation sets (validation size: {args.val_split:.1f})...")
    X = df['cleaned_review']
    y = df['polarity']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )
    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}")

    # Set Up Model
    model_builder = None
    model_params = {
        'embedding_dim': effective_embedding_dim,
        'dropout_rate': effective_dropout_rate,
        'maxlen': effective_maxlen
    }
    # Map model arg name to directory name (e.g., rnn_gru -> rnn-gru)
    model_dir_name = args.model
    if args.model == "rnn_gru":
        model_dir_name = "rnn-gru"
    elif args.model == "rnn_lstm":
        model_dir_name = "rnn-lstm"
    model_save_dir = MODELS_BASE_DIR / model_dir_name

    if args.model == "cnn":
        model_builder = build_cnn_model
        model_params['max_vocab'] = effective_vocab_size
        model_params['num_filters'] = cnn_filters
        model_params['kernel_size'] = cnn_kernel_size
    elif args.model == "rnn_gru":
        model_builder = build_gru_model
        model_params['max_vocab'] = effective_vocab_size
        model_params['gru_units'] = gru_units
    elif args.model == "rnn_lstm":
        model_builder = build_lstm_model
        model_params['vocab_size'] = effective_vocab_size
        model_params['lstm_units'] = lstm_units

    tokenizer_params = {'num_words': effective_vocab_size, 'oov_token': '<OOV>'}

    # Train
    print(f"\n--- Starting Training for {args.model.upper()} Model ---")
    if args.model == 'cnn':
        print(f"Using Hyperparameters: vocab={effective_vocab_size}, embed={effective_embedding_dim}, maxlen={effective_maxlen}, dropout={effective_dropout_rate}, lr={effective_lr}, filters={cnn_filters}, kernel={cnn_kernel_size}")
    elif args.model == 'rnn_gru':
        print(f"Using Hyperparameters: vocab={effective_vocab_size}, embed={effective_embedding_dim}, maxlen={effective_maxlen}, dropout={effective_dropout_rate}, lr={effective_lr}, units={gru_units}")
    elif args.model == 'rnn_lstm':
        print(f"Using Hyperparameters: vocab={effective_vocab_size}, embed={effective_embedding_dim}, maxlen={effective_maxlen}, dropout={effective_dropout_rate}, lr={effective_lr}, units={lstm_units}")

    model, history, tokenizer = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_builder=model_builder,
        model_params=model_params,
        tokenizer_params=tokenizer_params,
        maxlen=effective_maxlen,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=effective_lr,
        model_save_dir=model_save_dir,
        model_name=args.model_name
    )

    # Plot History
    print("\nPlotting training history...")
    plot_training_history(history)

    print(f"\n--- Training for {args.model.upper()} Model Finished Successfully! ---")


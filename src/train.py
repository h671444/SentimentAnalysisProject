import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from pandas import Series
from tensorflow.keras.models import Sequential

# assumes model builders are in src/models/
from models.cnn import build_cnn_model
# from models.rnn import build_rnn_model # example for future when i implement rnn

def train_model(X_train: Series, y_train: Series, X_val: Series, y_val: Series,
                model_builder: callable, 
                model_params: dict, 
                tokenizer_params: dict, 
                maxlen: int, 
                epochs: int = 20,
                batch_size: int = 64,
                learning_rate: float = 0.001,
                model_save_dir: str = '../models/trained', # More generic save dir
                model_name: str = 'best_model') -> Tuple[Sequential, Dict, Tokenizer]:

    print("Tokenizing data...")
    tokenizer = Tokenizer(**tokenizer_params)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)

    print(f"Padding sequences to maxlen={maxlen}...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=maxlen, padding='post')

    print("Building model...")
    # Pass maxlen explicitly if the builder needs it (adjust builder signatures accordingly)
    if 'maxlen' in model_builder.__code__.co_varnames:
        model = model_builder(**model_params, maxlen=maxlen)
    else:
         model = model_builder(**model_params)

    model.summary() # print model summary to console

    print("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # define save paths based on model_name
    run_save_dir = os.path.join(model_save_dir, model_name)
    os.makedirs(run_save_dir, exist_ok=True)
    model_checkpoint_path = os.path.join(run_save_dir, f'{model_name}.keras')
    tokenizer_save_path = os.path.join(run_save_dir, 'tokenizer.pickle')
    print(f"Model artifacts will be saved to: {run_save_dir}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print(f"Starting training (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val_pad, y_val),
                        callbacks=callbacks,
                        verbose=1) # Use verbose=1 for progress bars per epoch

    # save Tokenizer after training completes
    print(f"Saving tokenizer to {tokenizer_save_path}...")
    try:
        with open(tokenizer_save_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"ERROR: Failed to save tokenizer: {e}")
        # decide if failure to save tokenizer should stop everything

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


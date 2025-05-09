from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
)
from typing import List

def build_cnn_model(max_vocab: int,
                    embedding_dim: int,
                    maxlen: int,
                    dropout_rate: float,
                    num_filters: List[int],
                    kernel_size: int) -> Sequential:

    print(f"Building CNN: vocab={max_vocab}, embed_dim={embedding_dim}, maxlen={maxlen}, dropout={dropout_rate}")
    print(f"  Filters: {num_filters}, Kernel Size: {kernel_size}")

    if len(num_filters) < 3:
        raise ValueError("num_filters list must contain at least 3 values for the 3 Conv1D layers")

    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=embedding_dim, input_length=maxlen, name="embedding"),
        Dropout(dropout_rate, name="dropout_1"),
        Conv1D(filters=num_filters[0], kernel_size=kernel_size, activation='relu', name="conv1d_1"),
        MaxPooling1D(pool_size=2, name="maxpool_1"),
        Conv1D(filters=num_filters[1], kernel_size=kernel_size, activation='relu', name="conv1d_2"),
        MaxPooling1D(pool_size=2, name="maxpool_2"),
        Conv1D(filters=num_filters[2], kernel_size=kernel_size, activation='relu', name="conv1d_3"),
        MaxPooling1D(pool_size=2, name="maxpool_3"),
        GlobalMaxPooling1D(name="global_maxpool"),
        Dense(64, activation='relu', name="dense_1"),
        Dropout(dropout_rate, name="dropout_2"),
        Dense(1, activation='sigmoid', name="output")
    ], name="TextCNN")

    return model 
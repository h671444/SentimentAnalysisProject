from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional

def build_lstm_model(vocab_size: int,
                     embedding_dim: int,
                     maxlen: int,
                     dropout_rate: float,
                     lstm_units: int) -> Sequential:

    print(f"Building LSTM: vocab={vocab_size}, embed_dim={embedding_dim}, maxlen={maxlen}, dropout={dropout_rate}, lstm_units={lstm_units}")

    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=maxlen,
            mask_zero=True
        ),
        SpatialDropout1D(dropout_rate),
        Bidirectional(
            LSTM(
                units=lstm_units,
                dropout=dropout_rate,
                recurrent_dropout=0.0,
                return_sequences=False
            )
        ),
        Dense(1, activation='sigmoid')
    ], name="BiLSTM")

    return model

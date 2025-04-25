from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, SpatialDropout1D, Bidirectional

def build_gru_model(max_vocab: int,
                    embedding_dim: int,
                    maxlen: int,
                    dropout_rate: float,
                    gru_units: int) -> Sequential:

    print(f"Building GRU: vocab={max_vocab}, embed_dim={embedding_dim}, maxlen={maxlen}, dropout={dropout_rate}, gru_units={gru_units}")

    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim = embedding_dim, input_length = maxlen, name = "embedding"),
        SpatialDropout1D(dropout_rate),
        Bidirectional(
            GRU(
                units = gru_units,
                dropout = dropout_rate,
                return_sequences = False,
            )
        ),
        Dense(1, activation = "sigmoid"),
    ], name="BiGRU")

    return model 
    
    
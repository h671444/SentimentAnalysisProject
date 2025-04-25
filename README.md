# Sentiment Analysis with Deep Learning

## Project Overview

This project implements and compares different deep learning architectures for sentiment analysis. While our models are trained on Amazon product reviews, they can be applied to analyze the sentiment of any text. We've developed three distinct neural network models:

- **CNN (Convolutional Neural Network)**: Uses convolutional layers to extract features from text sequences
- **GRU (Gated Recurrent Unit)**: A simplified RNN variant that efficiently processes sequential data
- **LSTM (Long Short-Term Memory)**: An advanced RNN architecture designed to capture long-term dependencies

Each model was trained on a dataset of Amazon product reviews and evaluated using standard metrics like accuracy, precision, recall, and F1-score. The models can be used for binary sentiment classification (positive/negative) on any text input.

## Project Structure

- **app/**: Application code for deploying the models.
  - **Home.py**: Main application script for running the sentiment analysis app.
  - **components/**: UI components
- **data/**: Directory for storing raw and processed datasets.
  - **raw/**: Contains the original dataset files (e.g., `amazon_reviews.zip`).
  - **processed/**: Stores preprocessed data ready for model input.
- **models/**: Saved trained models (weights, checkpoints).
- **notebooks/**: Jupyter notebooks for exploration, development, and step-by-step training.
  - **cnn_nbs**: Notebooks related to the CNN model.
  - **rnn_nbs**: Notebooks related to the GRU and LSTM models.
- **reports/**: 
  - **SentimentAnalysisReport.pdf**: Detailed project report.
- **src/**: Structured, reusable, and production-quality code.
  - **preprocessing.py**: Data cleaning, preprocessing, feature engineering.
  - **models/**: Model definition scripts.
    - **cnn.py**: CNN architecture definition.
    - **rnn_gru.py**: BiGRU architecture definition.
    - **rnn_lstm.py**: BiLSTM architecture definition.
  - **train.py**: Main script for training the models.
  - **evaluate.py**: Script for evaluating trained models (metrics, confusion matrix).
- **.gitignore**: Standard Python gitignore file.
- **requirements.txt**: Lists all required Python libraries for a reproducible environment.
- **README.md**: Project documentation and setup instructions.

## Dataset

This project uses the Amazon Product Reviews dataset available on Kaggle.

1.  **Download:** Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews). It's typically a `.zip` file.
2.  **Placement:** Place the downloaded zip file (e.g., `amazon_reviews.zip`) inside the `data/raw/` directory.
3.  **Preprocessing:** The `src/preprocessing.py` script will handle extraction and processing, generating `data/processed/amazon_reviews_processed.csv`. The script assumes the raw CSV has columns like `polarity` (1 for negative, 2 for positive), `title`, and `text`. It combines title and text, cleans the text (lowercase, remove punctuation/numbers/stopwords), and prepares it for tokenization.

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Activate:
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Using Notebooks (Interactive Approach)

You can explore the data, preprocessing steps, model development, and training process interactively using the Jupyter notebooks provided in the `notebooks/` directory:

- `notebooks/cnn_nbs/`: Contains notebooks specific to the CNN model development and training.
- `notebooks/rnn_nbs/`: Contains notebooks for the BiGRU and BiLSTM models.

These notebooks provide a step-by-step walkthrough of the entire process and are useful for understanding the details or experimenting with different approaches. Ensure your environment (e.g., the activated `venv`) is available as a kernel in Jupyter.

### 2. Command-Line Interface (CLI)

Alternatively, you can use the following command-line tools to process data, train models, and evaluate results:

#### Preprocessing

Ensure the raw dataset is in `data/raw/`. Run the preprocessing script:
```bash
python src/preprocessing.py
```
This will create the processed CSV file in `data/processed/`.

#### Training

Train a specific model using the main training script. The script uses pre-defined optimal hyperparameters for each model type by default.
```bash
# Train the CNN model
python src/train.py --model cnn

# Train the BiGRU model
python src/train.py --model rnn_gru

# Train the BiLSTM model
python src/train.py --model rnn_lstm
```
Models and tokenizers will be saved in the corresponding subdirectory under `models/`.

You can override default hyperparameters via command-line arguments:
```bash
# Example: Train BiGRU with a different learning rate and more epochs
python src/train.py --model rnn_gru --lr 0.0005 --epochs 15
```
Common arguments include `--epochs`, `--batch_size`, `--maxlen`, `--vocab_size`, `--embedding_dim`, `--lr`, `--dropout_rate`, `--gru_units`, `--lstm_units`, `--cnn_kernel_size`.

#### Evaluation

Evaluate a trained model:
```bash
# Evaluate the saved CNN model
python src/evaluate.py --model cnn

# Evaluate the saved BiGRU model
python src/evaluate.py --model rnn_gru

# Evaluate the saved BiLSTM model
python src/evaluate.py --model rnn_lstm
```

### 3. Running the Application

To launch the Streamlit web interface for interactive sentiment analysis:
```bash
streamlit run app/Home.py
```

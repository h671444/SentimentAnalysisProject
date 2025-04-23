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
  - **components/**: UI components and utilities for the app interface.
  
- **data/**: Directory for storing raw and processed datasets.
  - **raw/**: Contains the original dataset files.
  - **processed/**: Stores preprocessed data ready for model input.

- **models/**: Saved trained models (weights, checkpoints).

- **notebooks/**: Exploratory analysis and development notebooks.
  - **cnn_nbs**: Initial data analysis and visualization.
  - **rnn_nbs**: Initial data analysis and visualization.

- **reports/**: 
  - **SentimentAnalysisReport.pdf**: Detailed project report.

- **src/**: Structured, reusable, and production-quality code.
  - **preprocessing.py**: Data cleaning, preprocessing, feature engineering.
  - **models/**: 
    - **cnn.py**: CNN implementation.
    - **rnn-gru.py**: RNN-GRU implementation.
    - **rnn-lstm.py**: RNN-LSTM implementation.
  - **train.py**: Main training pipeline with hyperparameter tuning support.
  - **evaluate.py**: Evaluation script (accuracy, confusion matrices, metrics).

- **requirements.txt**: Lists all required Python libraries for a reproducible environment.
- **.gitignore**: Standard Python gitignore file.

## Dataset

The dataset consists of Amazon product reviews, balanced with 50% positive and 50% negative sentiments. Preprocessing steps include text cleaning, tokenization, and stopword removal.

## Usage

To run the models, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/h671444/SentimentAnalysisProject.git
   cd SentimentAnalysisProject
   ```

2. **Set Up the Environment**:
   - Ensure you have Python 3.7 or later installed.
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Dataset**:
   - Place the raw dataset files in the `data/raw/` directory.
   - Run the preprocessing script to prepare the data:
     ```bash
     python src/preprocessing.py
     ```

5. **Train the Models**:
   - To train a specific model, use the training script with the desired model type:
     ```bash
     python src/train.py --model cnn  # Options: cnn, rnn-gru, rnn-lstm
     ```

6. **Evaluate the Models**:
   - Evaluate the trained models using the evaluation script:
     ```bash
     python src/evaluate.py --model cnn  # Options: cnn, rnn-gru, rnn-lstm
     ```

7. **Run the Application**:
   - If you have an application interface, run it using:
     ```bash
     streamlit run app/Home.py
     ```

8. **Additional Notes**:
   - Ensure all paths are correctly set in the scripts.
   - Modify hyperparameters and configurations as needed in the `train.py` script.
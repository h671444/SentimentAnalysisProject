import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
import zipfile
import pathlib

# Get the script's directory
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# Constants
RAW_DATA_DIR = SCRIPT_DIR / "../data/raw"
PROCESSED_DATA_DIR = SCRIPT_DIR / "../data/processed"
ZIP_FILENAME = "amazon_reviews.zip"
EXTRACTED_FOLDER_NAME = "amazon_reviews"
TRAIN_CSV_NAME = "train.csv"
# Define the single output filename
OUTPUT_PROCESSED_NAME = "amazon_reviews_processed.csv"
MAX_ROWS_TO_PROCESS = 200000

# attempt to download stopwords if not found
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def extract_zip(zip_path, extract_to_path):
    """Extracts a zip file if the target directory doesn't exist."""
    if not os.path.exists(extract_to_path):
        print(f"Extracting {zip_path} to {extract_to_path}...")
        os.makedirs(extract_to_path, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_path)
            print("Extraction complete.")
        except FileNotFoundError:
            print(f"ERROR: Zip file not found at {zip_path}")
            print("Please place amazon_reviews.zip in the data/raw directory.")
            raise
        except Exception as e:
            print(f"ERROR: Failed during extraction: {e}")
            raise
    else:
        print(f"Extraction directory {extract_to_path} already exists. Skipping extraction.")

def load_data(csv_path, max_rows=None):
    """Loads data from CSV, drops NA in title, converts polarity."""
    df = pd.read_csv(csv_path, header=None, names=["polarity", "title", "text"], nrows=max_rows)
    df.dropna(subset=["title"], inplace=True)
    # convert polarity (assumes 1 negative, 2 positive -> 0, 1)
    df['polarity'] = df['polarity'].apply(lambda x: 1 if x == 2 else 0)
    return df

def clean_text(text):
    """Lowercase, remove special chars, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    """Removes english stopwords from a space-separated string."""
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_data(df):
    """Combines title/text, cleans, removes stopwords."""
    # Keep original polarity for joining later
    df_orig = df[['polarity']].copy()
    df["full_review"] = df["title"] + " " + df["text"]
    df["cleaned_review"] = df["full_review"].apply(clean_text)
    df["cleaned_review"] = df["cleaned_review"].apply(remove_stopwords)
    print("Preprocessing complete.")
    # Return cleaned text and original index/polarity
    return df[['cleaned_review']].join(df_orig)

def save_data(df, output_path):
    """Saves DataFrame to a CSV file."""
    print(f"Saving data to {output_path}...")
    try:
        df.to_csv(output_path, index=False)
        print("Data saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save data to {output_path}: {e}")
        raise e


if __name__ == "__main__":
    # Ensure processed directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Extract
    zip_path = RAW_DATA_DIR / ZIP_FILENAME
    extract_folder_path = RAW_DATA_DIR / EXTRACTED_FOLDER_NAME
    extract_zip(str(zip_path), str(extract_folder_path))

    # Load
    csv_path = extract_folder_path / TRAIN_CSV_NAME
    if not csv_path.exists():
        print(f"ERROR: Expected CSV file not found after extraction: {csv_path}")
        exit(1)
    print(f"Loading data from {csv_path} (limit {MAX_ROWS_TO_PROCESS} rows)..." )
    df = load_data(str(csv_path), max_rows=MAX_ROWS_TO_PROCESS)

    # Preprocess
    print("Starting data preprocessing...")
    processed_df = preprocess_data(df)

    # Save
    output_path = PROCESSED_DATA_DIR / OUTPUT_PROCESSED_NAME
    save_data(processed_df[['cleaned_review', 'polarity']], str(output_path))

    print("\nPreprocessing script finished successfully!")

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# attempt to download stopwords if not found
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def load_data(csv_path, max_rows=None):

    df = pd.read_csv(csv_path, header=None, names=["polarity", "title", "text"], nrows=max_rows)
    initial_rows = len(df)

    df.dropna(subset=["title"], inplace=True)
    # convert polarity (assumes 1 negative, 2 positive -> 0, 1)
    df['polarity'] = df['polarity'].apply(lambda x: 1 if x == 2 else 0)

    return df

def clean_text(text):
    # lowercase, remove special chars, normalize whitespace
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    # removes english stopwords from a space-separated string
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def preprocess_data(df):
    # combines title/text, cleans, removes stopwords

    df["full_review"] = df["title"] + " " + df["text"]
    df["cleaned_review"] = df["full_review"].apply(clean_text)
    df["cleaned_review"] = df["cleaned_review"].apply(remove_stopwords)

    print("Preprocessing complete.")

    return df[['cleaned_review', 'polarity']]

def split_data(df, label_column='polarity', feature_column='cleaned_review',
               test_size=0.2, validation_size=0.2, random_state=42):
    X = df[feature_column]
    y = df[label_column]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Adjust validation size relative to the *remaining* train_val set size
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=y_train_val
    )
    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_data(df, output_path):
    # saves DataFrame to a CSV file
    print(f"Saving data to {output_path}...")
    try:
        df.to_csv(output_path, index=False)
        print("Data saved successfully.")
    except Exception as e:
        print(f"ERROR: Failed to save data to {output_path}: {e}")
        raise e

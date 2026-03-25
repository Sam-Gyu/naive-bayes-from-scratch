import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.data_loader import load_data
import os
import pickle

def initialize_nltk():
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for resource in resources:
        nltk.download(resource, quiet=True)

initialize_nltk()

ps = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))
if 'not' in STOPWORDS:
    STOPWORDS.remove('not')


def clean_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    cleaned_tokens = [ps.stem(word) for word in tokens if word not in STOPWORDS]
    return " ".join(cleaned_tokens)


def clean_df(df, column_name='cleaned_text', verbose=False):
    initial_count = len(df)
    df = df.dropna(subset=[column_name])
    df = df[df[column_name].str.strip() != ""]
    df = df.drop_duplicates(subset=[column_name])
    if verbose:
        final_count = len(df)
        print(f"Removed {initial_count - final_count} rows")
    return df

def vocabulary(text):
    vocab = {}
    index = 0

    for txt in text:
        words = txt.lower().split()
        for word in words:
            if word not in vocab:
                vocab[word] = index
                index += 1

    return vocab


def bag_of_word(text, vocab):
    bow_vectors = []

    for txt in text:
        # initialize vector with zeros
        vector = [0] * len(vocab)

        words = txt.lower().split()

        for word in words:
            if word in vocab:
                index = vocab[word]
                vector[index] += 1

        bow_vectors.append(vector)

    return bow_vectors


def vectorize_text(text_data, vocab=None):
    if vocab is None:
        vocab = vocabulary(text_data)
    features = bag_of_word(text_data, vocab)
    return features, vocab


def processed_data(path, sample_size=None, cache_path=None, vocab=None):
    if cache_path and os.path.exists(cache_path):
        print(f"--- Loading cached data from {cache_path} ---")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"--- Loading Data from {path} ---")
    df = load_data(path)
    print(f"Initial shape: {df.shape}")

    if sample_size:
        print(f"--- Using sample of {sample_size} ---")
        df = df.head(sample_size).copy()
        print(f"Shape after sampling: {df.shape}")

    print("--- Cleaning Data ---")
    df['cleaned_text'] = df['text'].apply(clean_text)
    df = clean_df(df, verbose=True)
    print(f"Shape after cleaning: {df.shape}")

    print("--- Vectorizing Data ---")
    features, vocab = vectorize_text(df['cleaned_text'], vocab=vocab)

    features = np.array(features, dtype=np.int32)
    features_df = pd.DataFrame(features, columns=list(vocab.keys()))

    target = df['target'].values

    result = (features_df, target, vocab)

    if cache_path:
        print(f"--- Saving processed data to {cache_path} ---")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)

    return result

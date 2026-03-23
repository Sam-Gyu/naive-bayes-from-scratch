import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.multinomial import MultinomialNB
from src.preprocessing.preprocessor import processed_data

if __name__ == "__main__":

    train_path = r"data\raw\aclImdb\train"
    test_path = r"data\raw\aclImdb\test"

    train_cache = r"data\cache\train_processed.pkl"
    test_cache = r"data\cache\test_processed.pkl"

    X_train, y_train, vectorizer = processed_data(
        train_path,
        cache_path=train_cache
    )

    X_test, y_test, _ = processed_data(
        test_path,
        vectorizer=vectorizer,
        cache_path=test_cache
    )

    print("--- Train Data ---")
    print(X_train.head())

    print("--- Test Data ---")
    print(X_test.head())

    model = MultinomialNB(alpha=1)
    model.fit(X_train, y_train)

    preds = model.predict_all(X_test)
    correct_test = sum(p == t for p, t in zip(preds, y_test))
    accuracy_test = correct_test / len(y_test)
    print(f"Accuracy Test: {accuracy_test:.4f}")
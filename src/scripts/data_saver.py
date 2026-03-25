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
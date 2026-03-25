from src.preprocessing.preprocessor import processed_data

def data_saver(train_path, test_path, train_cache, test_cache):
    X_train, y_train, vocab = processed_data(
        train_path,
        sample_size=10000,
        cache_path=train_cache
    )

    X_test, y_test, _ = processed_data(
        test_path,
        sample_size=2000,
        vocab=vocab,
        cache_path=test_cache
    )

    print("--- Train Data ---")
    print(X_train.head())
    print("--- Test Data ---")
    print(X_test.head())

    return X_train, y_train, X_test, y_test, vocab



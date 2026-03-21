import os
import pandas as pd

def load_data(data_dir):

    texts = []
    labels = []

    for sentiment in ['pos', 'neg']:
        folder = os.path.join(data_dir, sentiment)

        for file in os.listdir(folder):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1 if sentiment == 'pos' else 0)

    df = pd.DataFrame({
        "text": texts,
        "target": labels
    })

    return df
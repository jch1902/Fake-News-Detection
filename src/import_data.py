import os
import re
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    # Remove patterns at beginning of text such as "WASHINGTON (Reuters) -"
    cleaned_text = re.sub(r"^\S+(?:\s+\S+)*\s*\([^)]+\)\s*-", '', str(text))
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    return cleaned_text.lower()


def import_data():
    output_path = '../cleaned_data/final_labeled_fulldata.csv'
    if os.path.exists(output_path):
        print(output_path, "already exists!")
        df = pd.read_csv(output_path)
        df['text'] = df['text'].apply(clean_text)
        return df
    else:
        df = pd.read_csv('../WELFake_Dataset.csv')

        df['text'] = df['text'].apply(clean_text)
        df.dropna(subset=['title'], inplace=True)
        df.dropna(subset=['text'], inplace=True)
        df['fake_news_flag'] = df['label']
        df.drop(columns=["Unnamed: 0", "label"], inplace= True)
        true_df = pd.read_csv('../True.csv')
        false_df = pd.read_csv('../Fake.csv')
        
        true_df["fake_news_flag"] = 0

        true_df['text'] = true_df['text'].apply(clean_text)
        false_df["fake_news_flag"] = 1

        combined_df = pd.concat([true_df, false_df], ignore_index=True)
        df = pd.concat([combined_df, df], ignore_index=True)
        return df
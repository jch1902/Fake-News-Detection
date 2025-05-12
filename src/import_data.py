import os
import re
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    # Remove patterns at beginning of text such as "WASHINGTON (Reuters) -"
    cleaned_text = re.sub(r"^\S+(?:\s+\S+)*\s*\([^)]+\)\s*-", '', text)
    cleaned_text = re.sub(r"\n", "", cleaned_text) # removing \n
    cleaned_text = re.sub(r'\(.*?\)\s*-\s*', '', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    return cleaned_text


def import_data():
    output_path = '../cleaned_data/final_labeled_fulldata.csv'
    if os.path.exists(output_path):
        print(output_path, "already exists!")
        df = pd.read_csv(output_path)
        df['text'] = df['text'].apply(clean_text)
        return df
    else:
        # Combining true and false datasets
        true_df = pd.read_csv('../True.csv')
        false_df = pd.read_csv('../Fake.csv')
        
        true_df["fake_news_flag"] = 0
        false_df["fake_news_flag"] = 1

        combined_df = pd.concat([true_df, false_df], ignore_index=True)

        big_df = pd.read_csv("../WELFake_Dataset.csv")
        big_df["fake_news_flag"] = big_df["label"]

        big_df.drop(columns=["Unnamed: 0","label"], inplace= True)
        big_df = big_df.dropna()
        big_df.head()

        combined_df.drop(columns=['date','subject'], inplace= True)

        combined_df = pd.concat([combined_df, big_df], ignore_index=True)

        combined_df = shuffle(combined_df, random_state=42)



        # Preprocessing and vectorizing the text data
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(combined_df['text'])

        num_topics = 9  
        nmf_model = NMF(n_components=num_topics, random_state=42)
        nmf_model.fit(X)

        # Top words per topic
        terms = vectorizer.get_feature_names_out()
        topic_labels = []

        for topic_idx, topic in enumerate(nmf_model.components_):
            topic_words = [terms[i] for i in topic.argsort()[:-10 - 1:-1]] 
            print(f"Topic #{topic_idx}: {' '.join(topic_words)}")


        topic_label_map = {
            0: 'U.S. Politics',
            1: 'International Affairs',
            2: 'U.S. Politics',
            3: 'International Affairs',
            4: 'U.S. Politics',
            5: 'General News and Announcements',
            6: 'International Affairs',
            7: 'Legal and Political',
            8: 'Social Issue'
        }

        topic_assignments = nmf_model.transform(X)
        dominant_topic = topic_assignments.argmax(axis=1)

        combined_df['dominant_topic'] = dominant_topic
        combined_df['topic_label'] = combined_df['dominant_topic'].map(topic_label_map)
        combined_df.to_csv(output_path, index = False)
        combined_df['text'] = combined_df['text'].apply(clean_text)
        return combined_df
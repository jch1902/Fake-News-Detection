import pandas as pd
import keras
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from import_data import import_data
from import_data import clean_text


def split_data():
    df = import_data()

    X = df['text']  
    y = df['fake_news_flag']  


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=42)

    return X_train, X_test, y_train, y_test


def tokenize_data(tokenizer, data, max_length = 300):
    seq = tokenizer.texts_to_sequences(data)
    pad = pad_sequences(seq, maxlen=max_length, padding='post')
    return pad

def create_tokenizer(vocab_size = 10000):
    X_train, X_test, y_train, y_test = split_data()

    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    return tokenizer
    

def train_model(model, early_stop, vocab_size = 10000, max_length = 300, embedding_dim = 128):
    X_train, X_test, y_train, y_test = split_data()

    tokenizer = create_tokenizer

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
        
    model.fit(
            X_train_pad, y_train,
            epochs=5,
            batch_size=64,
            validation_data=(X_test_pad, y_test),
            callbacks=[early_stop]
        )
    
    return model, tokenizer


def fetch_model(vocab_size = 10000, max_length = 300, embedding_dim = 128):

    @register_keras_serializable()
    class Attention(Layer):
        def call(self, inputs):
            # inputs: (batch_size, time_steps, hidden_size)
            score = tf.nn.tanh(inputs)
            self.attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = self.attention_weights * inputs
            context_vector = tf.reduce_sum(context_vector, axis=1)
            return context_vector
        def get_attention_weights(self):
            return self.attention_weights

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Attention(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    

    # add early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',      
        patience=2,               
        restore_best_weights=True 
    )

    model_path = '../LSTMModel.keras'

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print("Succesfully loaded model!")
    else:
        model = train_model(model, early_stop)
        model.save('LSTMModel.keras')

    model.summary()

    return model

def predict(data, model = None):

    if model == None:
        model = fetch_model()



    data = clean_text(data)
    dataDF = pd.DataFrame()
    dataDF["text"] = [data]


    tokenizer = create_tokenizer()
    pad_data = tokenize_data(tokenizer, dataDF["text"])

    prediction_probs = model.predict(pad_data)
    predictions = (prediction_probs > 0.5).astype(int).flatten()
    return predictions





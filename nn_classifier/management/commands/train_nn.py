from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

def load_pickle_df(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
emotion_classes = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
]

def encode_labels_if_needed(labels):
    
    if isinstance(labels, pd.Series) or (isinstance(labels, np.ndarray) and labels.dtype == object):
        if isinstance(labels, np.ndarray):
            labels = pd.Series(labels.flatten())
        one_hot = pd.get_dummies(labels)
        one_hot = one_hot.reindex(columns=emotion_classes, fill_value=0)
        return one_hot.astype(np.float32).values
    elif isinstance(labels, np.ndarray) and np.issubdtype(labels.dtype, np.floating):
        return labels
    else:
        raise ValueError(f"Unexpected label data type or format: {type(labels)}, dtype: {getattr(labels, 'dtype', None)}")


class Command(BaseCommand):
    help = 'Train the neural network for emotion classification'

    def handle(self, *args, **options):
        print (' Loading dataset ')
        data_dir = r'\AI_final_project\ai_final_projetc\notebooks'
        train_path = os.path.join(data_dir, 'emotions_train.pkl')
        val_path = os.path.join(data_dir,'emotions_val.pkl')
        test_path = os.path.join(data_dir, 'emotions_test.pkl')

        df_train = load_pickle_df(train_path)
        df_val = load_pickle_df(val_path)
        df_test = load_pickle_df(test_path)

        texts_train = df_train['text'].values
        texts_val = df_val['text'].values
        texts_test = df_test['text'].values
        

        print('One hat encoding labesl....')
        labels_train = encode_labels_if_needed(df_train['predominant_emotion'])
        labels_val = encode_labels_if_needed(df_val['predominant_emotion'])
        labels_test = encode_labels_if_needed(df_test['predominant_emotion'])

        print(f"labels_train shape: {labels_train.shape}, dtype: {labels_train.dtype}")
        print(f"Sample labels_train row: {labels_train[0]}")

        max_words = 10000
        max_len = 50

        print('Fiting tokenizer on training and validation data')
        tokenzier = Tokenizer(num_words = max_words, oov_token= "<OOV>")
        tokenzier.fit_on_texts(np.concatenate([texts_train, texts_val]))

        print('Tokenizing and padding seqences....')
        seq_train = tokenzier.texts_to_sequences(texts_train)
        seq_val = tokenzier.texts_to_sequences(texts_val)
        seq_test = tokenzier.texts_to_sequences(texts_test)

        padded_train = pad_sequences(seq_train, maxlen= max_len, padding= 'post')
        padded_val = pad_sequences(seq_val, maxlen= max_len, padding='post')
        padded_test = pad_sequences(seq_test, maxlen= max_len, padding= 'post')

        keras_model_dir = os.path.join('nn_classifier', 'keras_model')
        os.makedirs(keras_model_dir, exist_ok= True)

        tokenzier_path = os.path.join(keras_model_dir, 'tokenizer.pkl')
        with open(tokenzier_path, 'wb') as f:
            pickle.dump(tokenzier, f)

        print('Building the neural model....')
        model = Sequential([
            Embedding(input_dim= max_words, output_dim= 128, input_length = max_len),
            GlobalAveragePooling1D(),
            Dense(64, activation= 'relu'),
            Dense(len(emotion_classes), activation= 'softmax')
        ])

        model.compile(
            loss= 'binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        checkpoint_path = os.path.join(keras_model_dir, 'best_model.keras')
        checkpoint =ModelCheckpoint(
            filepath= checkpoint_path,
            monitor= 'val_loss',
            save_best_only= True,
            mode ='min',
            verbose= 1
        )

        print('Training the model')
        model.fit(
            padded_train, labels_train,
            epochs= 100,
            batch_size= 32,
            validation_data= (padded_val, labels_val),
            callbacks= [checkpoint]
        )
        
        final_model_path = os.path.join(keras_model_dir, 'model.keras')
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
        os.rename(checkpoint_path, final_model_path)

        print(f'Training complete. Best model saved to {final_model_path}')

        print('Evaluating on test data')
        loss, acc = model.evaluate(padded_test, labels_test)
        print(f'Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')


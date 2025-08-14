
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

MODEL_PATH = r'\AI_final_project\nn_classifier\keras_model\model.keras'
TOKENZIZER_PATH = r'\AI_final_project\nn_classifier\keras_model\tokenizer.pkl'
MAX_LEN = 50

model = load_model(MODEL_PATH)
with open(TOKENZIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

EMOTIONS_CLASSES = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
]

def predict_emotion_nn(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen= MAX_LEN, padding= 'post')
    probs = model.predict(padded)[0]
    results_nn = []

    results_nn = [ {'label':  EMOTIONS_CLASSES[i], 'confidence': float(probs[i]) * 100}
               for i in range(len(probs)) if probs[i] > 0.075]
    results_nn.sort(key= lambda x: x['confidence'], reverse= True)
    print(results_nn)

    return results_nn

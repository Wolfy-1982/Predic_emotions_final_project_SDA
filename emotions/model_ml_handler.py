import os, joblib

BASE_DIR = os.path.dirname(__file__)
model_ml = joblib.load(os.path.join(BASE_DIR, r'ml_model/model.plk'))
vectorizer = joblib.load(os.path.join(BASE_DIR, r'ml_model/vectorizer.pkl'))
le = joblib.load(os.path.join(BASE_DIR,r'ml_model\label_encoder.pkl'))

def predict_emotion(text):
    vec = vectorizer.transform([text])
    pred_idx = model_ml.predict(vec)[0]
    prob = model_ml.predict_proba(vec).max()
    prob = float(prob)
    print( '*****************', prob, type(prob))
    return {
        'label': le.inverse_transform([pred_idx])[0],
        'confidence': prob
    }
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

emotions_train = pd.read_pickle(r'C:\AI_final_project\ai_final_projetc\notebooks\emotions_train.pkl')
emotions_val = pd.read_pickle(r'C:\AI_final_project\ai_final_projetc\notebooks\emotions_val.pkl')
emotions_test = pd.read_pickle(r'C:\AI_final_project\ai_final_projetc\notebooks\emotions_test.pkl')


x_train, y_train = emotions_train['text'], emotions_train['predominant_emotion']
x_val, y_val = emotions_val['text'], emotions_val['predominant_emotion']
x_test, y_test = emotions_test['text'], emotions_test['predominant_emotion']

vectorizer = TfidfVectorizer(max_features= 10000)
x_train_vec = vectorizer.fit_transform(x_train)
x_val_vec = vectorizer.fit_transform(x_val)
x_test_vec = vectorizer.fit_transform(x_test)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

model_ml = LogisticRegression(max_iter= 10000, class_weight= 'balanced')
model_ml.fit(x_train_vec, y_train_enc)

y_val_pred = model_ml.predict(x_val_vec)
print('Validation results: \n', classification_report(y_val_enc, y_val_pred, target_names= le.classes_))

y_test_pred = model_ml.predict(x_test_vec)
print('Test results:\n', classification_report(y_test_enc, y_test_pred, target_names= le.classes_))

joblib.dump(model_ml,r'emotions\ml_model\model.plk' )
joblib.dump(vectorizer, r'emotions\ml_model\vectorizer.pkl')
joblib.dump(le, r'emotions\ml_model\label_encoder.pkl')
from django.shortcuts import render
from .model_ml_handler import predict_emotion
from.predict_nn import predict_emotion_nn

def predict_view(request):
    prediction_ml = None
    prediction_nn = None

    if request.method == 'POST':
        text = request.POST.get('text', ' ')
        prediction_ml = predict_emotion(text)
        prediction_nn = predict_emotion_nn(text)
    
    return render(request, 'emotions/predict.html',
                  {'prediction_ml': prediction_ml,
                   'prediction_nn': prediction_nn})


# feature to work in parallel




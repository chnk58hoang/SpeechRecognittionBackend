from django.urls import path,include
from .views import *

urlpatterns = [
    path('predict/',Predict.as_view(),name='predict') #url http://127.0.0.1:8000/speech/predict/
]
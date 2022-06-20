from django.urls import path,include
from .views import *

urlpatterns = [
    path('predict/',Predict.as_view(),name='predict')
]
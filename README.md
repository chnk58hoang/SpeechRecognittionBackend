# SpeechRecognitionBackend

Backend server for SmartRoomSimulator. Developed with Django Rest Framework. Speech Recognition model was developed with TensorFlow.

# Requirements

If your device supports NVIDIA GPU:

```bash
pip install tensorflow
pip install tensorflow_text
pip install django
pip install djangorestframework
pip install django-cors-headers
```
Otherwise

```bash
pip install tensorflow_cpu
pip install tensorflow_text
pip install django
pip install djangorestframework
pip install django-cors-headers
```


# Load model from checkpoints

Our model was trained with GPU from Kaggle. Checkpoints were saved in /Speech/model_checkpoint and loaded for prediction task.
![image](https://user-images.githubusercontent.com/71833423/174621726-8ea5e631-c1e7-452f-bdec-7d0813fe1c7f.png)

# Prediction task

The commands are recored from frontend and sent to backend as tensors. Our model predicts label for each command and sents to frontend as strings. API url for this task (localhost): http://127.0.0.1:8000/speech/predict/
![image](https://user-images.githubusercontent.com/71833423/174623004-ebaeb4ef-4d3d-408b-9b9f-5a032fd86bd2.png)

# Run

```bash
python manage.py runserver
```

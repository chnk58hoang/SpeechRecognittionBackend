# SpeechRecognitionBackend

Backend server for SmartRoomSimulator. Developed with Django Rest Framework. Speech Recognition model is developed with TensorFlow.


# Requirements

```bash
pip install django
pip install djangorestframework
pip install tensorflow_text==2.8.2
```

# Load model from checkpoints
Our model was trained with GPU from Kaggle. The checkpoints were saved in /Speech/model_checkpoint and loaded for prediction task.
![image](https://user-images.githubusercontent.com/71833423/174621726-8ea5e631-c1e7-452f-bdec-7d0813fe1c7f.png)

# Prediction task
The commands are recored from frontend and sent to backend as tensors. Our model predicts label for each command and sents to frontend as strings.  
![image](https://user-images.githubusercontent.com/71833423/174623004-ebaeb4ef-4d3d-408b-9b9f-5a032fd86bd2.png)


# Run
```bash
python manage.py runserver
```

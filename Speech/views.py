from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .apps import SpeechConfig
from .serializers import *
import tensorflow as tf
import numpy as np
import librosa
import keras


# Create your views here.

class Predict(APIView):
    @staticmethod
    def extract_log_melspectrogram(waveform, sr=16000):
        stfts = tf.signal.stft(waveform, frame_length=1024, frame_step=256, fft_length=1024)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = tf.shape(spectrograms)[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 8000.0, 80

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate=sr,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz
        )

        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        log_mel_spectrogram = tf.math.log(mel_spectrograms + 1e-6)
        return log_mel_spectrogram

    @staticmethod
    def decode_batch_predictions(pred, lab=None):
        input_len = tf.ones(tf.shape(pred)[0], dtype=tf.int32) * tf.shape(pred)[1]
        # Use greedy search. For complex tasks, you can use beam search
        # results, _ = tf.nn.ctc_greedy_decoder(pred, lab, blank_index=pad_id)
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        results = tf.nn.relu(results)
        # Iterate over the results and get back the text
        output_text = []
        for result in results:
            result = SpeechConfig.tokenizer.detokenize(tf.cast(result, dtype=tf.int32)).numpy().decode('utf-8')
            output_text.append(result)
        return output_text

    @staticmethod
    def predict_label(log_mel_spectrogram):
        pred = SpeechConfig.model.predict(log_mel_spectrogram)
        return pred

    def post(self, request):
        audio = request.data['audio']
        audio = tf.cast(audio,dtype=tf.float32)
        #audio = tf.expand_dims(audio,axis=0)


        log_mel_spectrogram = self.extract_log_melspectrogram(waveform=audio,sr=16000)
        pred = self.predict_label(log_mel_spectrogram[tf.newaxis])
        decoded_pred = self.decode_batch_predictions(pred)

        prediction = " ".join(decoded_pred)

        return Response(prediction)

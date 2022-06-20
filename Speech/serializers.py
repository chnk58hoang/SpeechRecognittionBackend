from rest_framework import serializers


class PredictionResponse(serializers.Serializer):
    prediction = serializers.CharField(max_length=50)



class AudioRequest(serializers.Serializer):
    audio = serializers.ListField(child=serializers.DecimalField(max_digits=10,decimal_places=2))
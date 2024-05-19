from rest_framework import serializers

from .models import Event


class EventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = [
            "id",
            "event_name",
            "event_location",
            "event_date",
            "description",
            "image",
        ]

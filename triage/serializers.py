from rest_framework import serializers


class TicketSerializer(serializers.Serializer):

    title = serializers.CharField(
        max_length=255,
        required=True,
        allow_blank=False,
        trim_whitespace=True
    )

    description = serializers.CharField(
        max_length=5000,
        required=True,
        allow_blank=False,
        trim_whitespace=True
    )

    requester_department = serializers.CharField(
        max_length=100,
        required=True,
        allow_blank=False,
        trim_whitespace=True
    )
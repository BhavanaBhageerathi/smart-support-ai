import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import TicketSerializer
from .agent import run_ticket_triage


logger = logging.getLogger(__name__)


class TicketTriageView(APIView):

    def post(self, request):

        # Validate incoming ticket data
        serializer = TicketSerializer(data=request.data)

        if not serializer.is_valid():

            return Response(
                {
                    "error": "Invalid ticket details",
                    "details": serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get validated ticket details
        ticket = serializer.validated_data

        try:

            # Execute the AI ticket triage system
            result = run_ticket_triage(ticket)

            return Response(
                result,
                status=status.HTTP_200_OK
            )

        except Exception:

            # Record the actual error in the server logs
            logger.exception("Ticket triage processing failed")

            return Response(
                {
                    "error": "Unable to process the ticket at this time. Please try again later."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
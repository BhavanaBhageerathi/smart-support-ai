from django.urls import path
from .views import TicketTriageView

urlpatterns = [
    path("triage/", TicketTriageView.as_view(), name="ticket-triage"),
]
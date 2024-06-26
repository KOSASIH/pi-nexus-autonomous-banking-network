# pi_transaction_views.py
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import PITransaction
from .serializers import PITransactionSerializer

class PITransactionListView(APIView):
    def get(self, request):
        # ...

    def post(self, request):
        # ...

class PITransactionDetailView(APIView):
    def get(self, request, pk):
        # ...

    def put(self, request, pk):
        # ...

    def delete(self, request, pk):
        # ...

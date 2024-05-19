from django.db import models

class Ticket(models.Model):
    """
    Ticket model
    """
    ticket_id = models.AutoField(primary_key=True)
    user = models.ForeignKey('User', on_delete=models.CASCADE)
    event = models.ForeignKey('Event', on_delete=models.CASCADE)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.IntegerField()
    purchase_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=[
        ('available', 'Available'),
        ('sold', 'Sold'),
        ('cancelled', 'Cancelled')
    ])

    def __str__(self):
        return f"Ticket {self.ticket_id} for {self.event.event_name} - {self.quantity} x {self.price}"

    class Meta:
        unique_together = ('user', 'event')

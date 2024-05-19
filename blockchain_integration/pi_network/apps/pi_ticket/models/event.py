from django.db import models

class Event(models.Model):
    """
    Event model
    """
    event_id = models.AutoField(primary_key=True)
    event_name = models.CharField(max_length=100)
    event_location = models.CharField(max_length=100)
    event_date = models.DateTimeField()
    description = models.TextField()
    image = models.ImageField(upload_to='events/')

    def __str__(self):
        return self.event_name

    class Meta:
        ordering = ['event_date']

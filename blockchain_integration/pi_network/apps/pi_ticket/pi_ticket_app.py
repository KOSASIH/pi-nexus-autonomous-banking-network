from django.apps import AppConfig


class PiTicketAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pi_ticket_app"

    def ready(self):
        # Import any models, signals, or other code that needs to be initialized
        # when the app is loaded
        import pi_ticket_app.signals

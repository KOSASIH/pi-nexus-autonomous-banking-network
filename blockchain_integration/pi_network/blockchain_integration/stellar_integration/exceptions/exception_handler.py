# exception_handler.py
import requests
import json

class ExceptionHandler:
    def __init__(self, api_key, project_id):
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = f"https://your-error-reporting-api.com/api/{project_id}"

    def handle_exception(self, exception):
        data = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "stack_trace": self.get_stack_trace()
        }
        self.send_error_report(data)

    def send_error_report(self, data):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"Error reporting failed: {response.text}")

    def get_stack_trace(self):
        import traceback
        return traceback.format_exc()

# stellar_exception.py
import requests
import json
from stellar_sdk.exceptions import StellarSdkError

class StellarException(StellarSdkError):
    def __init__(self, message, code, data=None):
        super().__init__(message)
        self.code = code
        self.data = data
        self.api_key = "YOUR_ERROR_REPORTING_API_KEY"
        self.project_id = "YOUR_ERROR_REPORTING_PROJECT_ID"
        self.base_url = f"https://your-error-reporting-api.com/api/{self.project_id}"

    def __str__(self):
        return f"{self.code}: {self.message}"

    def report_error(self):
        data = {
            "exception_type": type(self).__name__,
            "exception_message": str(self),
            "stack_trace": self.get_stack_trace(),
            "data": self.data
        }
        self.send_error_report(data)

    def send_error_report(self, data):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code!= 200:
            print(f"Error reporting failed: {response.text}")

    def get_stack_trace(self):
        import traceback
        return traceback.format_exc()

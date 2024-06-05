import os

from google.cloud import functions


class GoogleCloudFunctions:

    def __init__(self, function_name):
        self.function_name = function_name
        self.client = functions.CloudFunctionsServiceClient()

    def invoke_function(self, event):
        # Invoke Google Cloud Function
        response = self.client.invoke_function(
            request={"name": self.function_name, "data": json.dumps(event)}
        )
        return response

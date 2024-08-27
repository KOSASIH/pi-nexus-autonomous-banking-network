import hashlib
import hmac

class DeviceAuth:
    def __init__(self, device_id, device_secret):
        self.device_id = device_id
        self.device_secret = device_secret

    def authenticate(self, request):
        # Generate a signature using the device secret and request data
        signature = hmac.new(self.device_secret.encode(), request.data, hashlib.sha256).hexdigest()
        # Verify the signature with the one provided in the request
        if signature == request.headers.get('X-Device-Signature'):
            return True
        return False

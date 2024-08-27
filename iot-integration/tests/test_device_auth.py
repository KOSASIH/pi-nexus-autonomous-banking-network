import unittest

from device_auth import DeviceAuth


class TestDeviceAuth(unittest.TestCase):

    def test_authenticate(self):
        device_auth = DeviceAuth("device_id", "device_secret")
        request = {"data": "some_data", "headers": {"X-Device-Signature": "signature"}}
        self.assertTrue(device_auth.authenticate(request))

    def test_authenticate_invalid_signature(self):
        device_auth = DeviceAuth("device_id", "device_secret")
        request = {
            "data": "some_data",
            "headers": {"X-Device-Signature": "invalid_signature"},
        }
        self.assertFalse(device_auth.authenticate(request))

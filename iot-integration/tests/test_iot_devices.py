import unittest
from iot_devices import SmartHomeDevice, WearableDevice, AutonomousVehicleDevice

class TestIotDevices(unittest.TestCase):
    def test_smart_home_device(self):
        device = SmartHomeDevice('SH123', 'device_token')
        self.assertEqual(device.get_utility_bill(), {'bill': 'some_bill'})

    def test_wearable_device(self):
        device = WearableDevice('WR456', 'device_token')
        self.assertEqual(device.authenticate_transaction('transaction_id'), {'authenticated': True})

    def test_autonomous_vehicle_device(self):
        device = AutonomousVehicleDevice('AV789', 'device_token')
        self.assertEqual(device.make_payment(10.99), {'payment': 'successful'})

class TestMicrotransactions(unittest.TestCase):
    def test_create_microtransaction(self):
        microtransaction_handler = MicrotransactionHandler('microtransactions.db')
        transaction_id = 'TX123'
        amount = 10.99
        microtransaction_handler.create_microtransaction(transaction_id, amount)
        self.assertTrue(microtransaction_handler.get_microtransaction(transaction_id))

    def test_get_microtransaction(self):
        microtransaction_handler = MicrotransactionHandler('microtransactions.db')
        transaction_id = 'TX123'
        amount = 10.99
        microtransaction_handler.create_microtransaction(transaction_id, amount)
        microtransaction = microtransaction_handler.get_microtransaction(transaction_id)
        self.assertEqual(microtransaction['amount'], amount)

class TestDataEncryption(unittest.TestCase):
    def test_encrypt_data(self):
        data = 'some_data'
        key = 'secret_key'
        encrypted_data = encrypt_data(data, key)
        self.assertNotEqual(encrypted_data, data)

    def test_decrypt_data(self):
        data = 'some_data'
        key = 'secret_key'
        encrypted_data = encrypt_data(data, key)
        decrypted_data = decrypt_data(encrypted_data, key)
        self.assertEqual(decrypted_data, data)

import edgeiq
from edgeiq import IoTDevice

class SupplyChainManagement:
    def __init__(self, iot_device):
        self.iot_device = iot_device

def register_device(self, device_data):
        self.iot_device.register(device_data)

    def update_status(self, device_id, status):
        self.iot_device.update_status(device_id, status)

# Example usage:
iot_device = IoTDevice()
supply_chain_manager = SupplyChainManagement(iot_device)
device_data = {'device_id': '12345', 'name': 'Sensor A', 'location': 'Warehouse 1'}
supply_chain_manager.register_device(device_data)
status = 'In Transit'
supply_chain_manager.update_status(device_data['device_id'], status)

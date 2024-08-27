Pi Nexus Autonomous Banking Network
=====================================

This repository contains the code for the Pi Nexus Autonomous Banking Network, a decentralized banking system that integrates with various IoT devices.

Getting Started
---------------

1. Install the required dependencies by running `pip install -r requirements.txt`
2. Run the Flask API by executing `python iot_api.py`
3. Use the IoT SDK to interact with the API

API Endpoints
-------------

* `/iot/devices`: Returns a list of supported IoT devices
* `/iot/devices/<device_type>`: Registers a new IoT device
* `/iot/transactions`: Makes a transaction using an IoT device

IoT Devices
------------

* Smart Home Device: `SmartHomeDevice` class
* Wearable Device: `WearableDevice` class
* Autonomous Vehicle Device: `AutonomousVehicleDevice` class

Microtransactions
----------------

* `MicrotransactionHandler` class: Handles microtransactions
* `create_microtransaction` method: Creates a new microtransaction
* `get_microtransaction` method: Retrieves a microtransaction

Data Encryption
----------------

* `encrypt_data` function: Encrypts data using a secret key
* `decrypt_data` function: Decrypts data using a secret key

Tests
-----

* `tests/test_device_auth.py`: Tests the device authentication module
* `tests/test_iot_devices.py`: Tests the IoT devices module
* `tests/test_microtransactions.py`: Tests the microtransactions module
* `tests/test_data_encryption.py`: Tests the data encryption module

def read_sensor_data(sensor_type, port):
    """
    Read data from a specific sensor.

    Args:
        sensor_type (str): The type of sensor (e.g., temperature, humidity).
        port (str): The port where the sensor is connected.

    Returns:
        dict: The sensor data.
    """
    if sensor_type == "temperature":
        sensor = TemperatureSensor(port)
        data = sensor.read_data()
    elif sensor_type == "humidity":
        sensor = HumiditySensor(port)
        data = sensor.read_data()
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    return process_sensor_data(data)


def process_sensor_data(data):
    """
    Process sensor data (e.g., clean, transform).

    Args:
        data (dict): The sensor data.

    Returns:
        dict: The processed data.
    """
    # Process data
    pass

import hardware


def read_temperature_sensor(i2c_address):
    """
    Reads the temperature from the specified I2C temperature sensor.
    """
    # Set up I2C communication
    sda_pin = 2
    scl_pin = 3
    hardware.setup_i2c(sda_pin, scl_pin)

    # Read temperature sensor data
    temperature_data = [0x00, 0x00]
    for i in range(2):
        hardware.write_i2c_data(i2c_address, [0x00], 1)
        temperature_data[i] = hardware.read_i2c_data(i2c_address, 1)[0]

    # Convert temperature data to temperature value
    temperature = (temperature_data[0] << 8) | temperature_data[1]
    temperature = temperature / 16.0

    # Clean up I2C communication
    hardware.cleanup()

    return temperature


def read_humidity_sensor(i2c_address):
    """
    Reads the humidity from the specified I2C humidity sensor.
    """
    # Set up I2C communication
    sda_pin = 2
    scl_pin = 3
    hardware.setup_i2c(sda_pin, scl_pin)

    # Read humidity sensor data
    humidity_data = [0x00, 0x00]
    for i in range(2):
        hardware.write_i2c_data(i2c_address, [0x00], 1)
        humidity_data[i] = hardware.read_i2c_data(i2c_address, 1)[0]

    # Convert humidity data to humidity value
    humidity = (humidity_data[0] << 8) | humidity_data[1]
    humidity = humidity / 16.0

    # Clean up I2C communication
    hardware.cleanup()

    return humidity

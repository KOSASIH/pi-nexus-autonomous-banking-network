# iot_integration/sensor.py
import Adafruit_DHT

# Set up a DHT11 temperature and humidity sensor
sensor = Adafruit_DHT.DHT11
pin = 4

# Read temperature and humidity values
humidity, temperature = Adafruit_DHT.read(sensor, pin)

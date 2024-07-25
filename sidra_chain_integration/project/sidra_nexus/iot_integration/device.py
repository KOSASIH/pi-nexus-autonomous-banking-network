# iot_integration/device.py
import RPi.GPIO as GPIO

# Set up a Raspberry Pi GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

# Turn on the pin
GPIO.output(17, GPIO.HIGH)

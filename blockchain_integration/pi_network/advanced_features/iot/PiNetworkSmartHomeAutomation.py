# Importing necessary libraries
import RPi.GPIO as GPIO
import time

# Class for smart home automation
class PiNetworkSmartHomeAutomation:
    def __init__(self):
        self.pin_map = {
            'living_room_light': 17,
            'kitchen_light': 23,
            'bedroom_light': 24,
            'fan': 25
        }
        GPIO.setmode(GPIO.BCM)

    # Function to turn on a device
    def turn_on(self, device):
        pin = self.pin_map[device]
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)

    # Function to turn off a device
    def turn_off(self, device):
        pin = self.pin_map[device]
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    # Function to toggle a device
    def toggle(self, device):
        pin = self.pin_map[device]
        GPIO.setup(pin, GPIO.OUT)
        current_state = GPIO.input(pin)
        GPIO.output(pin, not current_state)

# Example usage
automation = PiNetworkSmartHomeAutomation()
automation.turn_on('living_room_light')
time.sleep(5)
automation.turn_off('living_room_light')

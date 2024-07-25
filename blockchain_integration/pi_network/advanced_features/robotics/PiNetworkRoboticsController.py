# Importing necessary libraries
import RPi.GPIO as GPIO
import time

# Class for robotics controller
class PiNetworkRoboticsController:
    def __init__(self):
        self.pin_map = {
            'left_motor_forward': 17,
            'left_motor_backward': 23,
            'right_motor_forward': 24,
            'right_motor_backward': 25
        }
        GPIO.setmode(GPIO.BCM)

    # Function to move forward
    def move_forward(self):
        GPIO.output(self.pin_map['left_motor_forward'], GPIO.HIGH)
        GPIO.output(self.pin_map['right_motor_forward'], GPIO.HIGH)

    # Function to move backward
    def move_backward(self):
        GPIO.output(self.pin_map['left_motor_backward'], GPIO.HIGH)
        GPIO.output(self.pin_map['right_motor_backward'], GPIO.HIGH)

    # Function to turn left
    def turn_left(self):
        GPIO.output(self.pin_map['left_motor_backward'], GPIO.HIGH)
        GPIO.output(self.pin_map['right_motor_forward'], GPIO.HIGH)

    # Function to turn right
    def turn_right(self):
        GPIO.output(self.pin_map['left_motor_forward'], GPIO.HIGH)
        GPIO.output(self.pin_map['right_motor_backward'], GPIO.HIGH)

    # Function to stop
    def stop(self):
        GPIO.output(self.pin_map['left_motor_forward'], GPIO.LOW)
        GPIO.output(self.pin_map['left_motor_backward'], GPIO.LOW)
        GPIO.output(self.pin_map['right_motor_forward'], GPIO.LOW)
        GPIO.output(self.pin_map['right_motor_backward'], GPIO.LOW)

# Example usage
controller = PiNetworkRoboticsController()
controller.move_forward()
time.sleep(5)
controller.turn_left()
time.sleep(2)
controller.stop()

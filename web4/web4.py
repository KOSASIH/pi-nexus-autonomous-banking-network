import math

class Car:

    # Front end where user login is authenticated before starting the system
    def __init__(self, user, password, model, year, identification):
        self.user = user  # username
        self.password = password  # password
        self.model = model  # car model
        self.year = year  # year the car was made
        self.identification = identification  # user id
        # Welcome message
        print(f"Welcome {self.user} with ID {self.identification}. You are using car {self.model} from {self.year}.")

    # Operation 1: Car Control
    class Control:
        speed = 0  # in kilometers per hr (km/ph)
        max_speed = 120  # maximum speed before automatic braking
        car_on = True  # if car is started
        steer_direction = 0  # angle of steering
        car_controls = ("start", "steer", "accelerate", "brake", "stop", "update")  # a list of car controls
        real_time_data = []  # updates about the system will be listed here

        def start(self):
            if self.car_on:  # if true
                print(f"Car is starting.")
            else:  # if false
                print(f"Car is stopped.")

        def steer(self, steer_left, steer_right):
            if steer_left and steer_right:
                print("Error: Cannot steer left and right at the same time.")
            elif steer_left:
                self.steer_direction = -1
                print(f"Steering left.")
            elif steer_right:
                self.steer_direction = 1
                print(f"Steering right.")
            else:
                self.steer_direction = 0
                print(f"Steering straight.")

        def accelerate(self, speed):
            if speed >= 0:  # possible to speed up if speed is 0
                self.speed += speed
                print(f"Accelerating to {self.speed} km/ph.")
            else:
                print("Error: Cannot accelerate with a negative value.")

        def brake(self, speed):
            if speed >= 0:
                self.speed -= speed
                print(f"Braking to {self.speed} km/ph.")
            else:
                print("Error: Cannot brake with a negative value.")

        def stop(self):
            self.speed = 0
            print(f"Car has stopped.")

        def update(self):  # displays real time data about the car
            self.real_time_data = [self.speed, self.car_on, self.steer_direction]
            print(f"The state of the car is {self.real_time_data}.")

    # Operation 2: Environment Perception
    class Perception:
        present_location = []
        lane = True  # boolean value that states whether the car is driving within the lane
        avoid = ("other car", "humans", "animals", "trees", "plants")  # list of objects to avoid using LiDAR
        distance = 10  # in meters, maximum distance from object

        def detect_objects(self):
            # Uses LiDAR to detect objects and returns a list of objects
            pass

        def stay_in_lane(self):
            # Uses camera and machine learning to detect the lane and stay in it
            pass

    # Operation 3: Decision Making
    class Decision:
        def decide(self):
            # Makes decisions based on the perception and control
            pass

    # Operation 4: Car Execution
    class Execution:
        def execute(self):
            # Executes the decisions made
            pass

    # Front end where user login is authenticated before starting the system
    def login(self):
        # Authenticates user login
        pass

    # Operation 5: User Interface
    class UI:
        def display_data(self):
            # Displays real time data about the car
            pass

        def traffic_light(self):
            self.light = {}  # traffic light is detected and interpreted by the camera and machine learning
            if self.light == "red":  # stop light
                print(f"Car should stop.")  # car stops
            elif self.light == "yellow":  # slow down
                print(f"Car should slow down.")  # car slows down
            elif self.light == "green":  # go sign
                print(f"Car can move.")  # car drives

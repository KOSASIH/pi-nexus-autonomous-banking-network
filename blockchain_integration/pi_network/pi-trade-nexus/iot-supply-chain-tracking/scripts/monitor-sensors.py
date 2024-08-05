import csv
import time
from datetime import datetime

# Load sensor data from CSV file
sensor_data = []
with open('../data/sensor-data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        sensor_data.append(row)

# Define threshold values for sensor data
temperature_threshold = 25
humidity_threshold = 70
vibration_threshold = 1.0

# Monitor sensor data and alert if thresholds are exceeded
while True:
    for sensor in sensor_data:
        sensor_id = sensor['Sensor ID']
        shipment_id = sensor['Shipment ID']
        timestamp = datetime.strptime(sensor['Timestamp'], '%Y-%m-%d %H:%M:%S')
        temperature = float(sensor['Temperature (°C)'])
        humidity = float(sensor['Humidity (%)'])
        vibration = float(sensor['Vibration (g)'])

        # Check if sensor data exceeds thresholds
        if temperature > temperature_threshold:
            print(f'Alert: Temperature exceeded threshold for sensor {sensor_id} on shipment {shipment_id} at {timestamp}!')
        if humidity > humidity_threshold:
            print(f'Alert: Humidity exceeded threshold for sensor {sensor_id} on shipment {shipment_id} at {timestamp}!')
        if vibration > vibration_threshold:
            print(f'Alert: Vibration exceeded threshold for sensor {sensor_id} on shipment {shipment_id} at {timestamp}!')

    # Wait for 1 minute before checking sensor data again
    time.sleep(60)

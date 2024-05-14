import utils.math_utils as math_utils
import utils.network_utils as network_utils
import utils.sensor_utils as sensor_utils


def main():
    # Initialize sensors and network connections
    sensor_data = sensor_utils.read_sensor_data("temperature")
    processed_data = sensor_utils.process_sensor_data(sensor_data)
    network_utils.send_data_to_server(processed_data)


if __name__ == "__main__":
    main()

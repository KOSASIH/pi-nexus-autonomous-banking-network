import requests
import socket

def send_data_to_server(data, url):
    """
    Send data to a server using HTTP requests.

    Args:
        data (dict): The data to be sent.
        url (str): The URL of the server.

    Raises:
        ConnectionError: If the connection to the server fails.
    """
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to server: {e}")

def receive_data_from_server(url):
    """
    Receive data from a server using socket programming.

    Args:
        url (str): The URL of the server.

    Raises:
        ConnectionError: If the connection to the server fails.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((url, 80))
        data = sock.recv(1024)
        sock.close()
    except socket.error as e:
        print(f"Error receiving data from server: {e}")

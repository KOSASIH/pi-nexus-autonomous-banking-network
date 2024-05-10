import json
import requests
import websockets

def send_http_request(url, method, headers=None, data=None):
    """
    Sends an HTTP request to the specified URL using the specified method, headers, and data.
    """
    if headers is None:
        headers = {}

    if method == "GET":
        response = requests.get(url, headers=headers, data=data)
    elif method == "POST":
        response = requests.post(url, headers=headers, data=json.dumps(data))
    elif method == "PUT":
        response = requests.put(url, headers=headers, data=json.dumps(data))
    elif method == "DELETE":
response = requests.delete(url, headers=headers, data=data)
    else:
        raise ValueError("Invalid HTTP method")

    if response.status_code != 200:
        raise Exception("HTTP request failed with status code {}".format(response.status_code))

    return response.json()

def send_websocket_message(websocket, message):
    """
    Sends a message over a WebSocket connection.
    """
    websocket.send(json.dumps(message))

def receive_websocket_message(websocket):
    """
    Receives a message over a WebSocket connection.
    """
    return json.loads(websocket.recv())

def create_websocket_connection(url):
    """
    Creates a new WebSocket connection to the specified URL.
    """
    websocket = websockets.connect(url)
    return websocket

def close_websocket_connection(websocket):
    """
    Closes the specified WebSocket connection.
    """
    websocket.close()

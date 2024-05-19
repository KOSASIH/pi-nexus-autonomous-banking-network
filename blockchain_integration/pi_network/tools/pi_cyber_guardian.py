import re
import socket
from typing import List, Tuple

import requests


class PiCyberGuardian:
    def __init__(self, target_ip: str, target_port: int):
        self.target_ip = target_ip
        self.target_port = target_port

    def _send_request(
        self, request_type: str, request_data: bytes
    ) -> Tuple[bytes, bytes]:
        """Sends a request to the target IP and port and returns the response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.target_ip, self.target_port))
        sock.sendall(request_type.encode("utf-8") + request_data)
        response_type = sock.recv(1024).decode("utf-8")
        response_data = sock.recv(1024)
        sock.close()
        return response_type.encode("utf-8"), response_data

    def _parse_response(
        self, response_type: bytes, response_data: bytes
    ) -> Dict[str, Union[str, bool]]:
        """Parses the response from the target IP and port and returns a dictionary of response data."""
        response_data_str = response_data.decode("utf-8")
        response_data_dict = {}
        if response_type == b"GET":
            match = re.search(
                r"HTTP/1\.1 200 OK\r\nContent-Length: (\d+)\r\n\r\n(.+)",
                response_data_str,
            )
            if match:
                content_length = int(match.group(1))
                response_data_str = match.group(2)[:content_length]
                response_data_dict = {
                    "status_code": 200,
                    "body": response_data_str,
                }
            else:
                response_data_dict = {
                    "status_code": -1,
                    "body": response_data_str,
                }
        elif response_type == b"POST":
            match = re.search(r"HTTP/1\.1 (\d+) OK\r\n\r\n(.+)", response_data_str)
            if match:
                response_data_dict = {
                    "status_code": int(match.group(1)),
                    "body": match.group(2),
                }
            else:
                response_data_dict = {
                    "status_code": -1,
                    "body": response_data_str,
                }
        return response_data_dict

    def _check_for_vulnerabilities(
        self, response_data: Dict[str, Union[str, bool]]
    ) -> List[str]:
        """Checks the response data for vulnerabilities and returns a list of vulnerabilities found."""
        vulnerabilities = []
        if "HTTP/1.1 200 OK" not in response_data["body"]:
            vulnerabilities.append("HTTP response not 200 OK")
        if "Content-Security-Policy" not in response_data["body"]:
            vulnerabilities.append("Content Security Policy not set")
        if "X-Content-Type-Options" not in response_data["body"]:
            vulnerabilities.append("X-Content-Type-Options not set")
        if "X-Frame-Options" not in response_data["body"]:
            vulnerabilities.append("X-Frame-Options not set")
        if "X-XSS-Protection" not in response_data["body"]:
            vulnerabilities.append("X-XSS-Protection not set")
        if "Strict-Transport-Security" not in response_data["body"]:
            vulnerabilities.append("Strict-Transport-Security not set")
        if "Public-Key-Pins" not in response_data["body"]:
            vulnerabilities.append("Public-Key-Pins not set")
        if "Referrer-Policy" not in response_data["body"]:
            vulnerabilities.append("Referrer-Policy not set")
        if "Feature-Policy" not in response_data["body"]:
            vulnerabilities.append("Feature-Policy not set")
        return vulnerabilities

    def scan_target(self) -> List[str]:
        """Scans the target IP and port for vulnerabilities and returns a list of vulnerabilities found."""
        request_type = b"GET"
        request_data = b""
        response_type, response_data = self._send_request(request_type, request_data)
        response_data_dict = self._parse_response(response_type, response_data)
        vulnerabilities = self._check_for_vulnerabilities(response_data_dict)
        return vulnerabilities

import http.client

class XCMPConnector:
    def __init__(self, url: str, namespace: str):
        self.url = url
        self.namespace = namespace
        self.adapter = XCMPAdapter(namespace)

    def send_request(self, request: dict) -> dict:
        xcmp = self.adapter.convert_to_xcmp(request)
        headers = {"Content-Type": "application/xcmp+xml"}
        conn = http.client.HTTPConnection(self.url)
        conn.request("POST", "/", ET.tostring(xcmp, encoding="unicode"), headers)
        response = conn.getresponse()
        xcmp_response = ET.fromstring(response.read())
        return self.adapter.convert_from_xcmp(xcmp_response)

    def receive_request(self) -> dict:
        conn = http.client.HTTPConnection(self.url)
        conn.request("GET", "/")
        response = conn.getresponse()
        xcmp = ET.fromstring(response.read())
        return self.adapter.convert_from_xcmp(xcmp)

import influxdb

class InfluxDBClient:
    def __init__(self, host, port, username, password, database):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.client = influxdb.InfluxDBClient(host, port, username, password, database)

    def write_points(self, points):
        self.client.write_points(points)

    def query(self, query):
        return self.client.query(query)

influxdb_client = InfluxDBClient("localhost", 8086, "root", "root", "pi_network")
points = [
    {
        "measurement": "node_activity",
        "tags": {"node_id": "node-1"},
        "fields": {"activity": 10}
    }
]
influxdb_client.write_points(points)

result = influxdb_client.query("SELECT * FROM node_activity")
print(result)

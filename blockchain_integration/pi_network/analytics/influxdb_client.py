import influxdb

# Create an InfluxDB client
client = influxdb.InfluxDBClient(host="localhost", port=8086)

# Write data to InfluxDB
client.write_points(
    [
        {
            "measurement": "node_activity",
            "tags": {"node_id": "node-1"},
            "fields": {"activity": 10},
        }
    ]
)

# Query data from InfluxDB
result = client.query("SELECT * FROM node_activity")

import csv
import requests
from datetime import datetime

# Load shipment data from CSV file
shipment_data = []
with open('../data/shipment-data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        shipment_data.append(row)

# Define API endpoint for tracking shipments
api_endpoint = 'https://example.com/api/track-shipment'

# Track shipments and update status
for shipment in shipment_data:
    shipment_id = shipment['Shipment ID']
    origin = shipment['Origin']
    destination = shipment['Destination']
    departure_date = datetime.strptime(shipment['Departure Date'], '%Y-%m-%d')
    arrival_date = datetime.strptime(shipment['Arrival Date'], '%Y-%m-%d')

    # Send API request to track shipment
    response = requests.post(api_endpoint, json={
        'shipment_id': shipment_id,
        'origin': origin,
        'destination': destination,
        'departure_date': departure_date.isoformat(),
        'arrival_date': arrival_date.isoformat()
    })

    # Check if API request was successful
    if response.status_code == 200:
        print(f'Shipment {shipment_id} tracked successfully!')
    else:
        print(f'Error tracking shipment {shipment_id}: {response.text}')

    # Update shipment status in CSV file
    with open('../data/shipment-data.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Shipment ID', 'Origin', 'Destination', 'Departure Date', 'Arrival Date', 'Shipment Type', 'Weight (kg)', 'Volume (m³)'])
        writer.writeheader()
        for shipment in shipment_data:
            writer.writerow(shipment)

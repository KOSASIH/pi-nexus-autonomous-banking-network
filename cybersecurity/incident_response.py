import logging
import json

# Incident response logging
logging.basicConfig(filename='incident_response.log', level=logging.INFO)

# Function to log incident
def log_incident(incident_type, incident_data):
    logging.info(f'Incident detected: {incident_type} - {incident_data}')

# Function to respond to incident
def respond_to_incident(incident_type, incident_data):
    if incident_type == 'malware_detection':
        # Isolate infected system
        print('Isolating infected system...')
    elif incident_type == 'unauthorized_access':
        # Lock out unauthorized user
        print('Locking out unauthorized user...')
    else:
        print('Unknown incident type')

# Example usage
incident_type = 'malware_detection'
incident_data = {'system_ip': '192.168.1.100', 'malware_name': 'Trojan'}
log_incident(incident_type, incident_data)
respond_to_incident(incident_type, incident_data)

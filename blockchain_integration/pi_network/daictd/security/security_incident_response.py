import os
import json
from flask import Flask, request, jsonify
from security_oracles import SecurityOracleAPI

class SecurityIncidentResponse:
    def __init__(self, oracle_api):
        self.oracle_api = oracle_api

    def report_incident(self, incident_data):
        # Analyze incident data and determine response
        response = {'message': 'Incident reported successfully'}
        # Sign response with oracle
        response_signature = self.oracle_api.sign_data(json.dumps(response))
        return {'response': response, 'ignature': response_signature}

    def verify_response(self, response, signature):
        # Verify response signature with oracle
        if self.oracle_api.verify_signature(json.dumps(response), signature):
            return {'message': 'Response verified successfully'}
        else:
            return {'message': 'Invalid response signature'}, 401

class IncidentResponseAPI:
    def __init__(self, incident_response):
        self.incident_response = incident_response

    def report_incident(self):
        incident_data = request.json
        response = self.incident_response.report_incident(incident_data)
        return response

    def verify_response(self):
        response = request.json['response']
        signature = request.json['signature']
        return self.incident_response.verify_response(response, signature)

if __name__ == '__main__':
    oracle_api = SecurityOracleAPI(SecurityOracle('private_key.pem', 'public_key.pub'))
    incident_response = SecurityIncidentResponse(oracle_api)
    api = IncidentResponseAPI(incident_response)
    app = Flask(__name__)
    api.add_resource(api.report_incident, '/report_incident')
    api.add_resource(api.verify_response, '/verify_response')
    app.run(debug=True)

import os
import json
from flask import Flask, request, jsonify
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

class LoanProcessingAPI(Resource):
    def post(self):
        # Call the LoanProcessingContract's evaluateLoanApplication function
        response = requests.post(f'https://{os.environ["SIDRA_CHAIN_NETWORK"]}.sidra.chain/api/v1/contracts/LoanProcessingContract/evaluateLoanApplication', 
                                 headers={'Authorization': f'Bearer {os.environ["SIDRA_CHAIN_API_KEY"]}'},
                                 json=request.get_json())
        return response.json()

class InvestmentManagementAPI(Resource):
    def get(self):
        # Call the InvestmentManagementContract's analyzeMarketTrends function
        response = requests.get(f'https://{os.environ["SIDRA_CHAIN_NETWORK"]}.sidra.chain/api/v1/contracts/InvestmentManagementContract/analyzeMarketTrends', 
                                 headers={'Authorization': f'Bearer {os.environ["SIDRA_CHAIN_API_KEY"]}'})
        return response.json()

class RiskAssessmentAPI(Resource):
    def post(self):
        # Call the RiskAssessmentContract's identifyRisks function
        response = requests.post(f'https://{os.environ["SIDRA_CHAIN_NETWORK"]}.sidra.chain/api/v1/contracts/RiskAssessmentContract/identifyRisks', 
                                 headers={'Authorization': f'Bearer {os.environ["SIDRA_CHAIN_API_KEY"]}'},
                                 json=request.get_json())
        return response.json()

api.add_resource(LoanProcessingAPI, '/loan-processing')
api.add_resource(InvestmentManagementAPI, '/investment-management')
api.add_resource(RiskAssessmentAPI, '/risk-assessment')

if __name__ == '__main__':
    app.run(debug=True)

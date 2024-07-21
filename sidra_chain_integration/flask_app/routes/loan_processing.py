from flask import Blueprint, request, jsonify
from models import LoanApplication

bp = Blueprint('loan_processing', __name__)

@bp.route('/loan-processing', methods=['POST'])
def evaluate_loan_application():
    # Create a new loan application
    loan_application = LoanApplication(**request.get_json())
    db.session.add(loan_application)
    db.session.commit()

    # Call the LoanProcessingAPI
    response = requests.post('http://localhost:5000/loan-processing', json=request.get_json())
    return response.json()

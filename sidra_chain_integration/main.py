from sidra_chain_sdk import SidraChain
from loan_processing.loan_processing import LoanProcessing
from investment_management.investment_management import InvestmentManagement
from risk_assessment.risk_assessment import RiskAssessment

def main():
    # Initialize Sidra Chain SDK
    sidra_chain = SidraChain('YOUR_API_KEY', 'YOUR_API_SECRET', 'ainnet')

    # Create instances of autonomous banking use cases
    loan_processing = LoanProcessing(sidra_chain)
    investment_management = InvestmentManagement(sidra_chain)
    risk_assessment = RiskAssessment(sidra_chain)

    # Create and deploy smart contracts
    loan_processing.create_loan_processing_contract()
    investment_management.create_investment_management_contract()
    risk_assessment.create_risk_assessment_contract()

if __name__ == '__main__':
    main()

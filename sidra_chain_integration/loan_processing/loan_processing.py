from sidra_chain_sdk import SidraChain


class LoanProcessing:
    def __init__(self, sidra_chain):
        self.sidra_chain = sidra_chain

    def evaluate_loan_application(self, application):
        # Evaluate the loan application based on predefined criteria
        if (
            application["credit_score"] >= 700
            and application["income"] >= 50000
            and application["employment_history"] >= 2
            and application["loan_amount"] <= 10000
        ):
            return "APPROVED"
        else:
            return "REJECTED"

    def create_loan_processing_contract(self):
        # Create a smart contract for loan processing
        contract_name = "LoanProcessingContract"
        functions = [
            {
                "function": "evaluateLoanApplication",
                "inputs": ["application"],
                "outputs": ["decision"],
            }
        ]
        self.sidra_chain.create_smart_contract(contract_name, functions)
        self.sidra_chain.deploy_smart_contract(contract_name)

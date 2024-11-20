import json
from datetime import datetime

class SmartContract:
    def __init__(self, parties, terms):
        """
        Initialize a new smart contract.

        :param parties: List of parties involved in the contract.
        :param terms: Dictionary containing the terms of the contract.
        """
        self.parties = parties
        self.terms = terms
        self.is_executed = False
        self.execution_time = None
        self.events = []

    def execute(self):
        """
        Execute the smart contract if the terms are valid.
        """
        if self.is_valid():
            self.is_executed = True
            self.execution_time = datetime.now()
            self.log_event("Contract executed successfully.")
            return {
                "status": "success",
                "message": "Contract executed successfully.",
                "execution_time": self.execution_time.isoformat()
            }
        else:
            self.log_event("Contract execution failed due to invalid terms.")
            return {
                "status": "failure",
                "message": "Contract execution failed due to invalid terms."
            }

    def is_valid(self):
        """
        Validate the contract terms and parties.

        :return: True if valid, False otherwise.
        """
        # Example validation: Check if all required parties are present
        required_parties = self.terms.get('required_parties', [])
        return all(party in self.parties for party in required_parties)

    def log_event(self, message):
        """
        Log an event related to the contract.

        :param message: The message to log.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        self.events.append(event)

    def get_events(self):
        """
        Retrieve the event log for the contract.

        :return: List of events.
        """
        return self.events

    def to_dict(self):
        """
        Convert the smart contract to a dictionary representation.

        :return: Dictionary representation of the contract.
        """
        return {
            "parties": self.parties,
            "terms": self.terms,
            "is_executed": self.is_executed,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,
            "events": self.events
        }

    def to_json(self):
        """
        Convert the smart contract to a JSON string.

        :return: JSON string representation of the contract.
        """
        return json.dumps(self.to_dict(), indent=4)

# Example usage
if __name__ == "__main__":
    parties = ["Alice", "Bob"]
    terms = {
        "required_parties": ["Alice", "Bob"],
        "amount": 1000,
        "currency": "USD"
    }

    contract = SmartContract(parties, terms)
    print(contract.execute())  # Execute the contract
    print(contract.get_events())  # Get event log
    print(contract.to_json())  # Get JSON representation

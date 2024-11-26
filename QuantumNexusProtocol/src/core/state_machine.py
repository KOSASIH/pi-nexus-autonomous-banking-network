class StateMachine:
    def __init__(self):
        self.state = 'INITIAL'  # Initial state
        self.transitions = {
            'INITIAL': ['WAITING_FOR_TRANSACTION'],
            'WAITING_FOR_TRANSACTION': ['PROCESSING_TRANSACTION', 'MINING'],
            'PROCESSING_TRANSACTION': ['WAITING_FOR_TRANSACTION', 'MINING'],
            'MINING': ['WAITING_FOR_TRANSACTION', 'FINALIZED'],
            'FINALIZED': ['INITIAL']
        }

    def transition(self, event):
        """Handle state transitions based on events."""
        if self.state == 'INITIAL' and event == 'start':
            self.state = 'WAITING_FOR_TRANSACTION'
        elif self.state == 'WAITING_FOR_TRANSACTION' and event == 'transaction_received':
            self.state = 'PROCESSING_TRANSACTION'
        elif self.state == 'PROCESSING_TRANSACTION' and event == 'process_complete':
            self.state = 'MINING'
        elif self.state == 'MINING' and event == 'mining_complete':
            self.state = 'FINALIZED'
        elif self.state == 'FINALIZED' and event == 'reset':
            self.state = 'INITIAL'
        else:
            raise Exception(f"Invalid transition from {self.state} on event {event}")

    def get_state(self):
        """Return the current state."""
        return self.state

    def is_in_state(self, state):
        """Check if the state machine is in a specific state."""
        return self.state == state

    def available_transitions(self):
        """Return the available transitions from the current state."""
        return self.transitions.get(self.state, [])

# Example usage
if __name__ == '__main__':
    sm = StateMachine()
    print(f"Initial State: {sm.get_state()}")

    # Simulate state transitions
    try:
        sm.transition('start')
        print(f"State after starting: {sm.get_state()}")

        sm.transition('transaction_received')
        print(f"State after transaction received: {sm.get_state()}")

        sm.transition('process_complete')
        print(f"State after processing transaction: {sm.get_state()}")

        sm.transition('mining_complete')
        print(f"State after mining: {sm.get_state()}")

        sm.transition('reset')
        print(f"State after reset: {sm.get_state()}")
    except Exception as e:
        print(e)

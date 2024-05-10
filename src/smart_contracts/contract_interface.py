class ContractInterface:
    def deploy(self, initial_state: dict):
        pass

    def call(self, function_name: str, function_args: tuple):
        pass

    def handle_event(self, event_name: str, event_args: tuple):
        pass

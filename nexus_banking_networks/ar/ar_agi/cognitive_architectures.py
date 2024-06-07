import cognitive_architectures

class ARAGI:
    def __init__(self):
        self.cognitive_architectures = cognitive_architectures.CognitiveArchitectures()

    def enable_agi_based_decision_making(self, input_data):
        # Enable AGI-based decision making
        output = self.cognitive_architectures.process_input(input_data)
        return output

class AdvancedARAGI:
    def __init__(self, ar_agi):
        self.ar_agi = ar_agi

    def enable_cognitive_architecture_based_agi(self, input_data):
        # Enable cognitive architecture-based AGI
        output = self.ar_agi.enable_agi_based_decision_making(input_data)
        return output

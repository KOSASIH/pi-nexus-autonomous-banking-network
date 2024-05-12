import random


class SelfCorrector:
    def __init__(self, feedback_loop, transaction_parameters):
        self.feedback_loop = feedback_loop
        self.transaction_parameters = transaction_parameters

    def adjust_parameters(self, user_feedback=None, system_performance=None):
        """
        Adjusts the transaction parameters based on user feedback or system performance.
        """
        if user_feedback is not None:
            if user_feedback == "increase":
                self.feedback_loop.add_correction(
                    IncreaseParameterCorrection(self.transaction_parameters)
                )
            elif user_feedback == "decrease":
                self.feedback_loop.add_correction(
                    DecreaseParameterCorrection(self.transaction_parameters)
                )

        if system_performance is not None:
            if system_performance < 0.9:
                self.feedback_loop.add_correction(
                    IncreaseParameterCorrection(self.transaction_parameters)
                )
            elif system_performance > 0.95:
                self.feedback_loop.add_correction(
                    DecreaseParameterCorrection(self.transaction_parameters)
                )


class IncreaseParameterCorrection:
    def __init__(self, transaction_parameters):
        self.transaction_parameters = transaction_parameters

    def apply(self, transaction_parameters):
        """
        Increases the transaction parameter by a random amount.
        """
        parameter_name = random.choice(list(transaction_parameters.keys()))
        transaction_parameters[parameter_name] += random.uniform(0.1, 0.5)
        return transaction_parameters


class DecreaseParameterCorrection:
    def __init__(self, transaction_parameters):
        self.transaction_parameters = transaction_parameters

    def apply(self, transaction_parameters):
        """
        Decreases the transaction parameter by a random amount.
        """
        parameter_name = random.choice(list(transaction_parameters.keys()))
        transaction_parameters[parameter_name] -= random.uniform(0.1, 0.5)
        return transaction_parameters

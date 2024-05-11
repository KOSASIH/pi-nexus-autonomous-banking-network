import logging

class DecisionMaker:
    def __init__(self, rules, logger=None):
        self.rules = rules
        self.logger = logger or logging.getLogger(__name__)

    def make_decision(self, user, market_conditions, market_trend, transaction):
        """
        Makes a decision based on the defined decision rules.
        """
        self.logger.info("Making decision for transaction %s", transaction)
        for rule_name, rule_func in self.rules.items():
            if not rule_func(getattr(user, rule_name), transaction):
                self.logger.info("Rule %s failed for transaction %s", rule_name, transaction)
                return False
        self.logger.info("All rules passed for transaction %s", transaction)
        return True

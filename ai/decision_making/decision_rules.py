class DecisionRules:
    def __init__(self):
        self.rules = {
            'user_spending_limit': self.user_spending_limit_rule,
            'market_volatility': self.market_volatility_rule,
            'user_risk_tolerance': self.user_risk_tolerance_rule,
            'market_trend': self.market_trend_rule,
        }

    def user_spending_limit_rule(self, user, transaction):
        """
        Returns True if the transaction amount is less than or equal to the user's spending limit.
        """
        return transaction.amount <= user.spending_limit

    def market_volatility_rule(self, market_conditions, transaction):
        """
        Returns True if the market volatility is below a certain threshold.
        """
        return market_conditions.volatility < 0.1

    def user_risk_tolerance_rule(self, user, transaction):
        """
        Returns True if the user's risk tolerance is above a certain threshold.
        """
        return user.risk_tolerance > 0.5

    def market_trend_rule(self, market_trend, transaction):
        """
        Returns True if the market trend is positive.
        """
        return market_trend.direction == 'up'

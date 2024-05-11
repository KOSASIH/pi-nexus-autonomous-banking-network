class MarketConditionsAnalyzer:
    def __init__(self, data_preparation, time_series_analysis):
        self.data_preparation = data_preparation
        self.time_series_analysis = time_series_analysis

    def analyze_market_conditions(self, data_file, target_column, exogenous_variables, num_periods):
        """
        Analyzes market conditions using the time series analysis model.
        """
        data = self.data_preparation.prepare_data(data_file)

        model_fit = self.time_series_analysis.fit_model(data, target_column, exogenous_variables)
        market_trends = self.time_series_analysis.predict_market_trends(data, target_column, exogenous_variables, num_periods)

        return market_trends

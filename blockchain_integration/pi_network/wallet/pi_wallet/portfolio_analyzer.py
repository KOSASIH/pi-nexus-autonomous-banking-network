import pandas as pd
from portfolio_analysis import PortfolioAnalyzer
from models import ModelFactory

class PortfolioAnalyzerApp:
    def __init__(self, data_loader, model_factory):
        self.data_loader = data_loader
        self.model_factory = model_factory

    def analyze_portfolio(self, model_type):
        data = self.data_loader.load_data()
        analyzer = PortfolioAnalyzer()
        mse = analyzer.analyze_portfolio(data)
        print(f'Mean Squared Error: {mse}')

    def optimize_portfolio(self, model_type):
        data = self.data_loader.load_data()
        analyzer = PortfolioAnalyzer()
        model = self.model_factory.create_model(model_type)
        analyzer.optimize_portfolio(data, model)

if __name__ == '__main__':
    from data_loader import DataLoader
    data_loader = DataLoader()
    model_factory = ModelFactory()
    app = PortfolioAnalyzerApp(data_loader, model_factory)
    app.analyze_portfolio('linear')
    app.optimize_portfolio('linear')

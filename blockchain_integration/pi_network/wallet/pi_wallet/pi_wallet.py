import pandas as pd
import matplotlib.pyplot as plt

class SuperWallet:
    def __init__(self, user_data):
        self.user_data = user_data

    def get_portfolio_analysis(self):
        # Load user data into a pandas dataframe
        df = pd.DataFrame(self.user_data)

        # Calculate portfolio metrics (e.g. total value, returns, etc.)
        portfolio_metrics = df.groupby('asset').agg({'value': 'um', 'eturns': 'ean'})

        # Generate a report with visualizations
        report = """
        <h1>Portfolio Analysis</h1>
        <p>Total Value: ${:.2f}</p>
        <p>Average Returns: {:.2f}%</p>
        <img src="{}" alt="Portfolio Distribution">
        """.format(portfolio_metrics['value'].sum(), portfolio_metrics['returns'].mean() * 100, self.generate_portfolio_distribution_chart(df))

        return report

    def generate_portfolio_distribution_chart(self, df):
        # Generate a pie chart showing the distribution of assets in the portfolio
        plt.pie(df['value'], labels=df['asset'], autopct='%1.1f%%')
        plt.savefig('portfolio_distribution.png')
        return 'portfolio_distribution.png'

# Example usage
user_data = [
    {'asset': 'Pi Coin', 'value': 100, 'eturns': 0.05},
    {'asset': 'Bitcoin', 'value': 50, 'eturns': 0.03},
    {'asset': 'Ethereum', 'value': 20, 'eturns': 0.01}
]

super_wallet = SuperWallet(user_data)
print(super_wallet.get_portfolio_analysis())

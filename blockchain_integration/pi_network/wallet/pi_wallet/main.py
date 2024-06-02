import asyncio
from portfolio_analysis import PortfolioAnalyzer
from transaction_processor import TransactionProcessor
from data_loader import DataLoader
from security import SecurityManager
from models import ModelFactory

# Load user configuration
config = load_config('config.json')

# Set up security manager
security_manager = SecurityManager(config['user_id'], config['password'])

# Load user data
data_loader = DataLoader('user_data.csv')
portfolio = data_loader.load_data()

# Initialize model
model = ModelFactory('rf').create_model()

# Analyze portfolio
portfolio_analyzer = PortfolioAnalyzer(portfolio, model)
analysis = portfolio_analyzer.analyze_portfolio()

# Get recommendations
recommendations = portfolio_analyzer.get_recommendations()

# Set up transaction processor
transaction_processor = TransactionProcessor(None, None)

# Add new transactions to the transaction processor queue
for transaction in recommendations:
    transaction_processor.transaction_queue.put_nowait(transaction)

# Start processing transactions in the background
asyncio.run(transaction_processor.process_transactions())

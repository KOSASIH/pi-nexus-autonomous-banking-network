import pandas as pd
from ai_model_training.ai_model_architectures import CNNModel, RNNModel, TransformerModel
from ai_model_training.ai_model_training import train_ai_model
from ai_model_training.ai_model_deployment import deploy_ai_model

# Load the preprocessed financial data
df = pd.read_csv('data/preprocessed_data.csv')

# Train the AI models
cnn_model, cnn_history = train_ai_model(df, CNNModel)
rnn_model, rnn_history = train_ai_model(df, RNNModel)
transformer_model, transformer_history = train_ai_model(df, TransformerModel)

# Deploy the AI models to the cloud platform
deploy_ai_model('cnn_model', 'cloud_platform')
deploy_ai_model('rnn_model', 'cloud_platform')
deploy_ai_model('transformer_model', 'cloud_platform')

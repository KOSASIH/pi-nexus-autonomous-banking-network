import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html

class Visualizations:
    def __init__(self, data):
        self.data = data

    def create_dashboard(self):
        app = Dash(__name__)

        app.layout = html.Div([
            html.H1('Autonomous Banking Network Dashboard'),
            dcc.Graph(id='customer-behavior'),
            dcc.Graph(id='transaction-patterns'),
            dcc.Graph(id='network-performance')
        ])

        @app.callback(
            Output('customer-behavior', 'figure'),
            Input('data', 'value')
        )
        def update_customer_behavior(data):
            fig = px.scatter(data, x='age', y='income', color='segment')
            return fig

        @app.callback(
            Output('transaction-patterns', 'figure'),
            Input('data', 'value')
        )
        def update_transaction_patterns(data):
            fig = go.Bar(data, x='date', y='amount', color='category')
            return fig

        @app.callback(
            Output('network-performance', 'figure'),
            Input('data', 'value')
        )
        def update_network_performance(data):
            fig = go.Scatter3d(data, x='latency', y='throughput', z='availability', color='node')
            return fig

        app.run_server()

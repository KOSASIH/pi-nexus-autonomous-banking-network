# sidra_chain_dashboard.py
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sidra_chain_api import SidraChainAPI

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Sidra Chain Dashboard"),
        dcc.Graph(id="chain-data-graph"),
        dcc.Interval(id="interval-component", interval=1000),
    ]
)


@app.callback(
    Output("chain-data-graph", "figure"), [Input("interval-component", "n_intervals")]
)
def update_graph(n):
    # Retrieve chain data from the Sidra Chain API
    sidra_chain_api = SidraChainAPI()
    chain_data = sidra_chain_api.get_chain_data()
    # Process chain data using the Sidra Chain Data Processor
    data_processor = SidraChainDataProcessor(sidra_chain_api)
    predictions = data_processor.process_chain_data(chain_data)
    # Create a graph using the processed chain data
    fig = go.Figure(data=[go.Scatter(x=predictions.index, y=predictions.values)])
    return fig


if __name__ == "__main__":
    app.run_server()

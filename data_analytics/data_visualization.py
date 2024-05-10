import plotly.express as px

class DataVisualization:
    def __init__(self, data):
        self.data = data

    def visualize_data(self):
        # Create interactive dashboards
        fig = px.scatter(self.data, x='amount', y='frequency', color='cluster')
        fig.show()

        fig = px.histogram(self.data, x='transaction_type', nbins=20)
        fig.show()

        fig = px.box(self.data, x='cluster', y='amount')
        fig.show()

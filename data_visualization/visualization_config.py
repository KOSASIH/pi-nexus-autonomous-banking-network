class VisualizationConfig:
    def __init__(self, grafana_url, grafana_api_key):
        self.grafana_url = grafana_url
        self.grafana_api_key = grafana_api_key

    def get_data_visualizer(self):
        """
        Returns an instance of the DataVisualizer class.
        """
        return DataVisualizer(self.grafana_url, self.grafana_api_key)

import matplotlib.pyplot as plt
import seaborn as sns

class SidraChainDataVisualization:
    def __init__(self, connector):
        self.connector = connector

    def visualize_data_distribution(self, dataset_id):
        data_catalog = self.connector.get_data_catalog()
        dataset = data_catalog['datasets'][dataset_id]
        data = pd.DataFrame(dataset['data'])
        sns.distplot(data, kde=False)
        plt.title('Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    def visualize_data_correlation(self, dataset_id):
        data_catalog = self.connector.get_data_catalog()
        dataset = data_catalog['datasets'][dataset_id]
        data = pd.DataFrame(dataset['data'])
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
        plt.title('Data Correlation')
        plt.xlabel('Feature')
        plt.ylabel('Feature')
        plt.show()

    def visualize_data_lineage(self, dataset_id):
        data_lineage = self.connector.get_data_lineage(dataset_id)
        graph = nx.DiGraph()
        for node in data_lineage['nodes']:
            graph.add_node(node['id'], label=node['label'])
        for edge in data_lineage['edges']:
            graph.add_edge(edge['from'], edge['to'], label=edge['label'])
        nx.draw(graph, with_labels=True)
        plt.title('Data Lineage')
        plt.show()

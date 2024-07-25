import pandas as pd

class SidraChainDataIntegration:
    def __init__(self, connector):
        self.connector = connector

    def integrate_datasets(self, dataset_ids):
        datasets = []
        for dataset_id in dataset_ids:
            data_catalog = self.connector.get_data_catalog()
            dataset = data_catalog['datasets'][dataset_id]
            data = pd.DataFrame(dataset['data'])
            datasets.append(data)
        integrated_data = pd.concat(datasets, axis=0)
        return integrated_data

    def integrate_data_with_external_source(self, dataset_id, external_data):
        data_catalog = self.connector.get_data_catalog()
        dataset = data_catalog['datasets'][dataset_id]
        data = pd.DataFrame(dataset['data'])
        integrated_data = pd.merge(data, external_data, on='common_column')
        return integrated_data

    def apply_data_transformation(self, dataset_id, transformation_type):
        data_catalog = self.connector.get_data_catalog()
        dataset = data_catalog['datasets'][dataset_id]
        data = pd.DataFrame(dataset['data'])
        if transformation_type == 'log':
            data['value'] = data['value'].apply(lambda x: np.log(x))
        elif transformation_type == 'normalize':
            data['value'] = data['value'].apply(lambda x: x / x.max())
        return data

    def apply_data_filtering(self, dataset_id, filter_type, filter_value):
        data_catalog = self.connector.get_data_catalog()
        dataset = data_catalog['datasets'][dataset_id]
        data = pd.DataFrame(dataset['data'])
        if filter_type == 'greater_than':
            data = data[data['value'] > filter_value]
        elif filter_type == 'less_than':
            data = data[data['value'] < filter_value]
        return data

    def apply_data_aggregation(self, dataset_id, aggregation_type):
        data_catalog = self.connector.get_data_catalog()
        dataset = data_catalog['datasets'][dataset_id]
        data = pd.DataFrame(dataset['data'])
        if aggregation_type == 'um':
            data = data.groupby('column').sum()
        elif aggregation_type == 'ean':
            data = data.groupby('column').mean()
        return data

import pandas as pd

class SidraChainDataProfiling:
    def __init__(self, connector):
        self.connector = connector

    def get_data_profile(self, dataset_id):
        data_catalog = self.connector.get_data_catalog()
        dataset = data_catalog['datasets'][dataset_id]
        data = pd.DataFrame(dataset['data'])
        profile = data.profile_report()
        return profile

    def get_data_quality_score(self, dataset_id):
        data_quality = self.connector.get_data_quality(dataset_id)
        score = 0
        for metric in data_quality['metrics']:
            score += metric['value']
        return score / len(data_quality['metrics'])

    def get_data_complexity_score(self, dataset_id):
        data_lineage = self.connector.get_data_lineage(dataset_id)
        score = 0
        for node in data_lineage['nodes']:
            score += len(node['dependencies'])
        return score / len(data_lineage['nodes'])

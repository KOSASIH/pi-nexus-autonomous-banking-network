import os
import requests

class DataVisualizer:
    def __init__(self, grafana_url, grafana_api_key):
        self.grafana_url = grafana_url
        self.grafana_api_key = grafana_api_key

    def create_dashboard(self, dashboard_title, panels):
        """
        Creates a new dashboard in Grafana.
        """
        headers = {
            'Authorization': f'Bearer {self.grafana_api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self.grafana_url}/api/dashboards/db'
        data = {
            'dashboard': {
                'title': dashboard_title,
                'panels': panels
            }
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f'Failed to create dashboard: {response.text}')

        return response.json()['dashboard']['uid']

    def update_dashboard(self, dashboard_uid, panels):
        """
        Updates an existing dashboard in Grafana.
        """
        headers = {
            'Authorization': f'Bearer {self.grafana_api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self.grafana_url}/api/dashboards/uid/{dashboard_uid}'
        data = {
            'dashboard': {
                'panels': panels
            }
        }

        response = requests.put(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f'Failed to update dashboard: {response.text}')

        return response.json()['dashboard']['uid']

    def delete_dashboard(self, dashboard_uid):
        """
        Deletes a dashboard in Grafana.
        """
        headers = {
            'Authorization': f'Bearer {self.grafana_api_key}'
        }

        url = f'{self.grafana_url}/api/dashboards/uid/{dashboard_uid}'

        response = requests.delete(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f'Failed to delete dashboard: {response.text}')

        return response.json()['message']

    def create_panel(self, panel_title, panel_type, panel_json):
        """
        Creates a new panel in Grafana.
        """
        headers = {
            'Authorization': f'Bearer {self.grafana_api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self.grafana_url}/api/panels/database'
        data = {
            'title': panel_title,
            'type': panel_type,
            'json': panel_json
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f'Failed to create panel: {response.text}')

        return response.json()['panelId']

    def update_panel(self, panel_id, panel_json):
        """
        Updates an existing panel in Grafana.
        """
        headers = {
            'Authorization': f'Bearer {self.grafana_api_key}',
            'Content-Type': 'application/json'
        }

        url = f'{self.grafana_url}/api/panels/{panel_id}'
        data = {
            'json': panel_json
        }

        response = requests.put(url, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f'Failed to update panel: {response.text}')

        return response.json()['panelId']

    def delete_panel(self, panel_id):
        """
        Deletes a panel in Grafana.
        """
        headers = {
            'Authorization': f'Bearer {self.grafana_api_key}'
        }

        url = f'{self.grafana_url}/api/panels/{panel_id}'

        response = requests.delete(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f'Failed to delete panel: {response.text}')

        return response.json()['message']

import json
import requests

def generate_report():
    response = requests.get("http://localhost:8545/report")
    if response.status_code == 200:
        report_data = response .json()
        with open('network_report.json', 'w') as f:
            json.dump(report_data, f, indent=4)
        print("Report generated successfully: network_report.json")
    else:
        print("Error generating report:", response.status_code)

if __name__ == "__main__":
    generate_report()

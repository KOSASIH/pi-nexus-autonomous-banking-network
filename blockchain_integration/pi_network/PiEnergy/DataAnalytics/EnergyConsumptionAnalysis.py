import pandas as pd

def analyze_energy_consumption(data):
    # Load energy consumption data
    consumption_data = pd.read_csv(data)

    # Calculate total energy consumption
    total_consumption = consumption_data['consumption'].sum()

    # Calculate average energy consumption
    average_consumption = total_consumption / len(consumption_data)

    # Return analysis results
    return {
        'total_consumption': total_consumption,
        'average_consumption': average_consumption
    }

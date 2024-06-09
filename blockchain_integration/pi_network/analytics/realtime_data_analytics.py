import pandas as pd
import matplotlib.pyplot as plt

def analyze_data(data):
    # Real-time data analytics implementation
    df = pd.DataFrame(data)
    plt.plot(df['timestamp'], df['value'])
    plt.show()
    return df

# dex_data_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns

class DexDataVisualizer:
    def __init__(self):
        pass

    def visualize_dex_data(self, data):
        # Visualize DEX data
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='timestamp', y='value', data=df)
        plt.title('DEX Data Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.show()

    def visualize_dex_data_distribution(self, data):
        # Visualize DEX data distribution
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.distplot(df['value'], kde=False)
        plt.title('DEX Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

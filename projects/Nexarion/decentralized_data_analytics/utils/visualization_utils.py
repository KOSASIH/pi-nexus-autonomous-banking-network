import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(data, title):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='date', y='value', data=data)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()

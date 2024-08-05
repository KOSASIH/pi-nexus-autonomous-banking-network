import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EonixDataAnalysis:
    def __init__(self):
        self.df = None

    def load_data(self, df):
        self.df = df

    def summary_stats(self):
        print(self.df.describe())

    def correlation_matrix(self):
        corr_matrix = self.df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
        plt.show()

    def histogram(self, column):
        plt.hist(self.df[column], bins=50)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def scatter_plot(self, x, y):
        plt.scatter(self.df[x], self.df[y])
        plt.title(f'Scatter Plot of {x} vs {y}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def box_plot(self, column):
        plt.boxplot(self.df[column])
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.show()

    def missing_value_analysis(self):
        missing_values = self.df.isnull().sum()
        print("Missing Values:")
        print(missing_values)

    def outlier_detection(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = self.df[~((self.df[column] >= Q1 - 1.5 * IQR) & (self.df[column] <= Q3 + 1.5 * IQR))]
        print("Outliers:")
        print(outliers)

    def data_profiling(self):
        profile = self.df.profile_report()
        print("Data Profile:")
        print(profile)

    def feature_importance(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature Importances:")
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

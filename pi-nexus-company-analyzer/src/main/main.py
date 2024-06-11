# main.py

import os
import json
from nlp import NLP
from ml import ML
from visualization import Visualization

class Main:
    def __init__(self):
        self.nlp = NLP('bert-base-uncased', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.ml = ML('random_forest', 'data.csv')
        self.visualization = Visualization('github_api_data.json')

    def run_nlp(self):
        self.nlp.load_data('github_api_data.json')
        self.nlp.preprocess_data()
        self.nlp.train_model()
        self.nlp.evaluate_model()

    def run_ml(self):
        self.ml.train_model()
        self.ml.evaluate_model()

    def run_visualization(self):
        self.visualization.plot_confusion_matrix(self.ml.y_test, self.ml.y_pred)

if __name__ == '__main__':
    main = Main()
    main.run_nlp()
    main.run_ml()
    main.run_visualization()

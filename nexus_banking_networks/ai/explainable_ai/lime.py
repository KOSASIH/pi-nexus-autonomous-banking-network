import lime
from lime.lime_tabular import LimeTabularExplainer

class LIMEExplainer:
    def __init__(self, model, data, num_features):
        self.model = model
        self.data = data
        self.num_features = num_features
        self.explainer = LimeTabularExplainer(data, feature_names=data.columns, class_names=['positive', 'negative'])

    def explain_instance(self, instance):
        exp = self.explainer.explain_instance(instance, self.model.predict_proba, num_features=self.num_features)
        return exp

class ModelInterpreter:
    def __init__(self, model):
        self.model = model

    def interpret(self, instance):
        explainer = LIMEExplainer(self.model, instance, 5)
        explanation = explainer.explain_instance(instance)
        return explanation

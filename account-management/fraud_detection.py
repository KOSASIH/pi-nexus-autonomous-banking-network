import tensorflow as tf
from graph_sage import GraphSAGE

class FraudDetection:
    def __init__(self, graph_sage_model: GraphSAGE, tensorflow_model: tf.keras.Model):
        self.graph_sage_model = graph_sage_model
        self.tensorflow_model = tensorflow_model

    def detect_fraud(self, transaction_data: list) -> list:
        # Analyze transaction patterns using deep learning
        predictions = self.tensorflow_model.predict(transaction_data)
        # Model account relationships using graph neural networks
        graph_embeddings = self.graph_sage_model.embeddings(transaction_data)
        # Detect suspicious behavior using explainable AI
        explanations = self.explain_fraud(predictions, graph_embeddings)
        return explanations

    def explain_fraud(self, predictions: list, graph_embeddings: list) -> list:
        # Use LIME to explain fraud detection results
        explanations = []
        for i, prediction in enumerate(predictions):
            explanation = lime.explain_instance(prediction, graph_embeddings[i])
            explanations.append(explanation)
        return explanations

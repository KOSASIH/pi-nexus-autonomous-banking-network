# services/fraud_detection_service.py
import tensorflow as tf


class FraudDetectionService:
    def __init__(self, model):
        self.model = model

    def detect_fraud(self, transaction):
        input_data = tf.convert_to_tensor([transaction])
        prediction = self.model.predict(input_data)
        if prediction > 0.5:
            return True
        else:
            return False

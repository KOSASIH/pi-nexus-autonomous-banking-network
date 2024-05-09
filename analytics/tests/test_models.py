import unittest
from unittest.mock import patch
from models.model_training import ModelTraining
from models.model_evaluation import ModelEvaluation
from models.model_deployment import ModelDeployment

class TestModels(unittest.TestCase):
    @patch('models.model_training.ModelTraining.train')
    @patch('models.model_evaluation.ModelEvaluation.evaluate')
    @patch('models.model_deployment.ModelDeployment.deploy')
    def test_run_pipeline(self, mock_deploy, mock_evaluate, mock_train):
        model_training = ModelTraining("test_model", None)
        model_evaluation = ModelEvaluation(None, None)
        model_deployment = ModelDeployment(None, None)
        models = Models()
        models.run_pipeline(model_training, model_evaluation, model_deployment)
        mock_train.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_deploy.assert_called_once()

if __name__ == '__main__':
    unittest.main()

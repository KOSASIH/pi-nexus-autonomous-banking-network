import time


class Learner:
    def __init__(self, data_preparation, model_training, model_evaluation):
        self.data_preparation = data_preparation
        self.model_training = model_training
        self.model_evaluation = model_evaluation

    def learn_and_improve(
        self, data_file, target_column, model_file, improvement_threshold
    ):
        """
        Learns and improves the machine learning model using the prepared data.
        """
        while True:
            data = self.data_preparation.prepare_data()
            model = self.model_training.train_model(model_file)
            rmse = self.model_evaluation.evaluate_model(data, target_column)

            print(f"Current RMSE: {rmse}")

            if rmse < improvement_threshold:
                print("Improvement threshold reached. Exiting.")
                break

            time.sleep(60)

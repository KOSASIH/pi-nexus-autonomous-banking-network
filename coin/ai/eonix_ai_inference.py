import tensorflow as tf
from tensorflow import keras
from eonix_ai_models import EonixAIModel, EonixAIModelV2, EonixAIModelV3
from eonix_ai_data_preprocessing import preprocess_data

def load_model(model_path):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file

    Returns:
        model (keras.Model): The loaded model
    """
    model = keras.models.load_model(model_path)
    return model

def make_prediction(model, input_data):
    """
    Make a prediction using the model.

    Args:
        model (keras.Model): The trained model
        input_data (pd.DataFrame): The input data

    Returns:
        prediction (np.ndarray): The predicted output
    """
    prediction = model.predict(input_data)
    return prediction

def get_class_label(prediction, class_labels):
    """
    Get the class label from the prediction.

    Args:
        prediction (np.ndarray): The predicted output
        class_labels (list): List of class labels

    Returns:
        class_label (str): The predicted class label
    """
    class_label = class_labels[np.argmax(prediction)]
    return class_label

def main():
    # Load the model
    model_path = 'best_model.h5'
    model = load_model(model_path)

    # Load the input data
    input_data = pd.read_csv('input_data.csv')

    # Preprocess the input data
    input_data_preprocessed = preprocess_data(input_data, categorical_cols=['category'], numerical_cols=['feature1', 'feature2'], text_cols=['text'])

    # Make a prediction
    prediction = make_prediction(model, input_data_preprocessed)

    # Get the class label
    class_labels = ['class1', 'class2', 'class3', 'class4', 'class5']
    class_label = get_class_label(prediction, class_labels)

    # Print the result
    print(f'Predicted class label: {class_label}')

if __name__ == '__main__':
    main()

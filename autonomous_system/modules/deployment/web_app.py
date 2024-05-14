from flask import Flask, jsonify, request
from model_evaluation import evaluate_model

from config import get_config

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    # Load transformed data from CSV file
    input_file = get_config("data_processing.input_file")
    df = pd.read_csv(input_file)

    # Extract features from the request data
    feature1 = request.json["feature1"]
    feature2 = request.json["feature2"]
    feature3 = request.json["feature3"]

    # Create a new DataFrame with the input data
    new_data = pd.DataFrame(
        [{"feature1": feature1, "feature2": feature2, "feature3": feature3}]
    )

    # Load the trained model from the pickle file
    output_file = get_config("model_training.output_file")
    model = LogisticRegression.load(output_file)

    # Make a prediction using the model
    y_pred = model.predict(new_data)

    # Return the prediction as JSON
    return jsonify({"prediction": y_pred[0]})


if __name__ == "__main__":
    app.run(debug=True)

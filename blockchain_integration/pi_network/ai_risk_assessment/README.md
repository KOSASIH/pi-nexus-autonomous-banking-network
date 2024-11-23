# AI Risk Assessment

This directory contains the implementation of an AI model for risk assessment and a service for integrating the model with blockchain technology.

## Directory Structure

- `risk_assessment_model.py`: AI model for assessing risk using machine learning.
- `risk_assessment_service.py`: Flask service for integrating the AI model with blockchain.
- `README.md`: Documentation for the AI Risk Assessment project.

## Risk Assessment Model (`risk_assessment_model.py`)

The `RiskAssessmentModel` class implements a Random Forest classifier for risk assessment.

### Functions

- `train(data_path)`: Trains the model using the dataset located at `data_path`.
- `save_model(model_path)`: Saves the trained model to the specified path.
- `load_model(model_path)`: Loads a pre-trained model from the specified path.
- `predict(input_data)`: Predicts the risk label for the given input features.

### Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib

### Installation

1. Install the required packages:
   ```bash
   pip install pandas scikit-learn joblib
   ```

2. Train the model using your dataset:
   ```python
   1 model = RiskAssessmentModel()
   2 model.train('path_to_your_dataset.csv')
   3 model.save_model('risk_assessment_model.pkl')
   ```

# Risk Assessment Service (risk_assessment_service.py)
The Risk Assessment Service is a Flask application that provides an endpoint for assessing risk using the AI model.

# Endpoints

   - **POST** /assess_risk: Accepts a JSON body with features (list of input features) and returns the risk prediction.

## Requirements

   - Flask
   - Web3.py

## Installation
Install the required packages:

   ```bash
   1 pip install Flask web3
   ```

## Run the service:

   ```bash
   1 python risk_assessment_service.py
   ```

## License
This project is licensed under the MIT License.

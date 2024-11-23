from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# Sample data generation for demonstration
def generate_sample_data():
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'value': np.random.randint(1, 100, size=100)
    }
    return pd.DataFrame(data)

# Load or generate data
data = generate_sample_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    # Convert DataFrame to JSON
    return jsonify(data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

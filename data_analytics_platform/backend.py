import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import create_engine

app = Flask(__name__)
CORS(app)

# Database connection
engine = create_engine('postgresql://user:password@host:port/dbname')

# Load data from database
def load_data():
    data = pd.read_sql_table('pi_coin_data', engine)
    return data

# API endpoint to retrieve real-time data
@app.route('/api/data', methods=['GET'])
def get_data():
    data = load_data()
    return jsonify(data.to_dict(orient='records'))

# API endpoint to retrieve aggregated data
@app.route('/api/aggregated_data', methods=['GET'])
def get_aggregated_data():
    data = load_data()
    aggregated_data = data.groupby('date').agg({'price': 'mean', 'volume': 'sum'})
    return jsonify(aggregated_data.to_dict(orient='records'))

# API endpoint to retrieve listing progress data
@app.route('/api/listing_progress', methods=['GET'])
def get_listing_progress():
    data = load_data()
    listing_progress = data.groupby('listing_status').size().reset_index(name='count')
    return jsonify(listing_progress.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)

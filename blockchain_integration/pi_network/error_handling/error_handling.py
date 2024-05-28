# error_handling.py

import os
import logging
import logging.config
from flask import Flask, request, jsonify
from flask.logging import default_handler

app = Flask(__name__)

# Logging configuration
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'default'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 1000000,
            'backupCount': 3,
            'level': 'INFO',
            'formatter': 'default'
        }
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        }
    }
})

logger = logging.getLogger(__name__)

@app.errorhandler(404)
def not_found_error(error):
    logger.error(f"404 Error: {error}")
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"500 Error: {error}")
    return jsonify({"error": "Internal Server Error"}), 500

@app.route("/api/data", methods=["GET"])
def get_data():
    try:
        # Data retrieval logic here
        data = []
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        return jsonify({"error": "Error retrieving data"}), 500

@app.route("/api/process", methods=["POST"])
def process_data():
    try:
        # Data processing logic here
        result = {}
        return jsonify(result)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return jsonify({"error": "Invalid input"}), 400
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return jsonify({"error": "Error processing data"}), 500

if __name__ == "__main__":
    app.run(debug=True)

# log_analysis.py

import os
import logging
import logging.handlers
from datetime import datetime

def analyze_logs(log_file):
    log_data = []
    with open(log_file, 'r') as f:
        for line in f:
            log_data.append(line.strip())
    
    error_count = 0
    warning_count = 0
    info_count = 0
    debug_count = 0
    
    for log in log_data:
        if log.startswith('[ERROR]'):
            error_count += 1
        elif log.startswith('[WARNING]'):
            warning_count += 1
        elif log.startswith('[INFO]'):
            info_count += 1
        elif log.startswith('[DEBUG]'):
            debug_count += 1
    
    print(f"Error count: {error_count}")
    print(f"Warning count: {warning_count}")
    print(f"Info count: {info_count}")
    print(f"Debug count: {debug_count}")

if __name__ == "__main__":
    log_file = 'app.log'
    analyze_logs(log_file)

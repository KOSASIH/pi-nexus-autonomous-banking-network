#!/bin/bash

# Start the application
gunicorn -b 0.0.0.0:8000 app:app

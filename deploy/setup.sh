#!/bin/bash

# Create the database
export FLASK_APP=app.py
flask db init
flask db migrate -m "Initial migration"
flask db upgrade

# Create the application directory
sudo mkdir /var/www/pi-nexus-banking

# Copy the application files to the application directory
sudo cp * /var/www/pi-nexus-banking

# Set the application owner and permissions
sudo chown -R pi:pi /var/www/pi-nexus-banking
sudo chmod -R 755 /var/www/pi-nexus-banking

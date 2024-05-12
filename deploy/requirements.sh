#!/bin/bash

# Install Python 3.x
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# Install Flask and other dependencies
pip3 install -r requirements.txt

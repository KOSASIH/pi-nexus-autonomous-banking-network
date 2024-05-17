#!/bin/bash

# Set the working directory
cd /path/to/pi_coin_auto_lister

# Start the pi\_coin\_stable\_value\_updater.py script in the background
python pi_coin_stable_value_updater.py &

# Start the pi\_coin\_auto\_lister.py script
python pi_coin_auto_lister.py

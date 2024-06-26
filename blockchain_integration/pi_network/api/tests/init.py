# Initialize the test suite
import os
import sys
import unittest

# Set up the test environment
os.environ['TESTING'] = 'True'

# Import the API modules for testing
from pi_network.api import app, models, routes

# Load the test configuration
from .config import TEST_CONFIG

# Set up the test database
from pi_network.api.models import db
db.init_app(app, TEST_CONFIG)

# Load the test fixtures
from .fixtures import load_fixtures

# Set up the test client
from pi_network.api.routes import api
test_client = api.test_client()

# Define the test suite
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestApp))
    suite.addTest(unittest.makeSuite(TestModels))
    suite.addTest(unittest.makeSuite(TestRoutes))
    return suite

# Run the test suite
if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

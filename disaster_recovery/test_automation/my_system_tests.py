import unittest
import requests

# Set up test cases
class TestMySystem(unittest.TestCase):
    """Test cases for the MySystem class."""

    def setUp(self):
        """Set up test environment."""
        self.base_url = 'http://localhost:8000'

    def test_get_status(self):
        """Test the get_status method of the MySystem class."""
        url = f'{self.base_url}/status'
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

    def test_get_data(self):
        """Test the get_data method of the MySystem class."""
        url = f'{self.base_url}/data'
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()), 0)

    def test_post_data(self):
        """Test the post_data method of the MySystem class."""
        url = f'{self.base_url}/data'
        data = {'key': 'value'}
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {'message': 'Data received'})

# Run test cases
if __name__ == '__main__':
    unittest.main()

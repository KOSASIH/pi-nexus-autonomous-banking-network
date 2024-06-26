import unittest
from pi_network.api import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_app_config(self):
        self.assertEqual(app.config['TESTING'], True)
        self.assertEqual(app.config['DEBUG'], False)

    def test_app_routes(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_app_healthcheck(self):
        response = self.app.get('/healthcheck')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

if __name__ == '__main__':
    unittest.main()

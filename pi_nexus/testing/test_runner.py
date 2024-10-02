# testing/test_runner.py
import unittest


class TestRunner:
    def __init__(self):
        self.test_cases = [
            # implementation
        ]

    def run_tests(self):
        suite = unittest.TestSuite()
        for test_case in self.test_cases:
            suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(test_case))
        runner = unittest.TextTestRunner()
        runner.run(suite)

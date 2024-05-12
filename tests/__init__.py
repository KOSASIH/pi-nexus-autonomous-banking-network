import unittest

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromModule(test_block))
    suite.addTest(loader.loadTestsFromModule(test_blockchain))
    suite.addTest(loader.loadTestsFromModule(test_transaction))
    suite.addTest(loader.loadTestsFromModule(test_wallet))
    suite.addTest(loader.loadTestsFromModule(test_node))
    suite.addTest(loader.loadTestsFromModule(test_p2p_network))
    return suite

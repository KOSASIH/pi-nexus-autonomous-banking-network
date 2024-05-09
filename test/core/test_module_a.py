import unittest
import core.module_a

class TestModuleA(unittest.TestCase):
    def test_method_a(self):
        module = core.module_a.ModuleA()
        result = module.method_a()
        self.assertEqual(result, 'expected result')

    def test_method_b(self):module = core.module_a.ModuleA()
        result = module.method_b()
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()

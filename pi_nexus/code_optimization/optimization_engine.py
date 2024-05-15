# code_optimization/optimization_engine.py
import ast


class OptimizationEngine:
    def __init__(self):
        self.optimization_rules = OptimizationRules()

    def optimize_code(self, code):
        tree = ast.parse(code)
        optimized_tree = self.apply_optimization_rules(tree)
        optimized_code = compile(optimized_tree, "<string>", "exec")
        return optimized_code

    def apply_optimization_rules(self, tree):
        optimized_tree = copy.deepcopy(tree)
        for rule in self.optimization_rules.rules:
            rule.apply(optimized_tree)
        return optimized_tree

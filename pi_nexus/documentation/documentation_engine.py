# documentation/documentation_engine.py
import ast
import inspect


class DocumentationEngine:
    def __init__(self):
        self.documentation_rules = DocumentationRules()

    def generate_documentation(self, code):
        tree = ast.parse(code)
        documentation = self.apply_documentation_rules(tree)
        return documentation

    def apply_documentation_rules(self, tree):
        documentation = {}
        for rule in self.documentation_rules.rules:
            documentation.update(rule.apply(tree))
        return documentation

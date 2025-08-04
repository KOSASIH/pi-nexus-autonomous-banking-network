# code_security/security_engine.py
import ast


class SecurityEngine:
    def __init__(self):
        self.security_rules = SecurityRules()

    def audit_code(self, code):
        tree = ast.parse(code)
        vulnerabilities = self.apply_security_rules(tree)
        return vulnerabilities

    def apply_security_rules(self, tree):
        vulnerabilities = []
        for rule in self.security_rules.rules:
            vulnerabilities += rule.apply(tree)
        return vulnerabilities

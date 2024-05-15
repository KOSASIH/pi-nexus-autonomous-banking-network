# code_review/review_engine.py
import ast

class ReviewEngine:
    def __init__(self):
        self.review_rules = ReviewRules()

    def review_code(self, code):
        tree = ast.parse(code)
        issues = self.apply_review_rules(tree)
        return issues

    def apply_review_rules(self, tree):
        issues = []
        for rule in self.review_rules.rules:
            issues += rule.apply(tree)
        return issues

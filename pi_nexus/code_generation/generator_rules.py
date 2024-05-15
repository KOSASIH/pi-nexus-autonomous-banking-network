# code_generation/generator_rules.py
class GeneratorRules:
    def __init__(self):
        self.templates = [
            # implementation
        ]

    def get_templates(self, requirements):
        selected_templates = []
        for template in self.templates:
            if template.matches(requirements):
                selected_templates.append(template)
        return selected_templates

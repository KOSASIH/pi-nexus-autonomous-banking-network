# code_generation/generator_engine.py
import inspect


class GeneratorEngine:
    def __init__(self):
        self.generator_rules = GeneratorRules()

    def generate_code(self, requirements):
        templates = self.generator_rules.get_templates(requirements)
        generated_code = self.generate_from_templates(templates)
        return generated_code

    def generate_from_templates(self, templates):
        generated_code = ""
        for template in templates:
            generated_code += template.generate()
        return generated_code

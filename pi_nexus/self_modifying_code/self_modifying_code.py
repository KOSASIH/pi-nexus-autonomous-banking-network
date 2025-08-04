import ast


class SelfModifyingCode:
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)

    def modify_code(self, new_code: str) -> str:
        self.tree.body[0].value = ast.parse(new_code).body[0].value
        return ast.unparse(self.tree)

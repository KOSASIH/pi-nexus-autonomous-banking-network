import ast
from refactor import RefactoringTool

class CodeRefactorer:
    def __init__(self, code: str):
        self.code = code
        self.tree = ast.parse(code)

    def refactor_code(self) -> str:
        tool = RefactoringTool(self.tree)
        tool.apply_refactorings()
        return ast.unparse(self.tree)

from models.ai_model import AIModel

class AIService:
    def __init__(self, ai_model: AIModel):
        self.ai_model = ai_model

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return self.ai_model.predict(data)

    def evaluate(self, data: pd.DataFrame) -> float:
        return self.ai_model.evaluate(data)

    def train(self, data: pd.DataFrame) -> None:
        self.ai_model.train(data)

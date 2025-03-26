import ast

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CodeReviewer:
    def __init__(self, model_name: str, tokenizer_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def review_code(self, code: str) -> dict:
        inputs = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt",
        )
        outputs = self.model(
            inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        scores = outputs.logits.detach().numpy()
        return {"code_quality": scores[0][0], "code_security": scores[0][1]}

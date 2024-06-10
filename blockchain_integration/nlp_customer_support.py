import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer

class NLPCustomerSupport:
    def __init__(self, model_name: str):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def generate_response(self, customer_query: str) -> str:
        input_text = f"translate English to Spanish: {customer_query}"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0])
        return response

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

class AdvancedNLP:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def tokenize_text(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return inputs

    def get_embeddings(self, text):
        inputs = self.tokenize_text(text)
        outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def classify_text(self, text):
        embeddings = self.get_embeddings(text)
        outputs = torch.nn.functional.softmax(embeddings, dim=1)
        _, predicted = torch.max(outputs, dim=1)
        return predicted

    def generate_text(self, prompt):
        inputs = self.tokenize_text(prompt)
        outputs = self.model.generate(inputs['input_ids'], max_length=50)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Example usage:
advanced_nlp = AdvancedNLP('bert-base-uncased')

text = 'This is a sample text.'
embeddings = advanced_nlp.get_embeddings(text)
print(embeddings.shape)

classified_text = advanced_nlp.classify_text(text)
print(classified_text)

prompt = 'This is a sample prompt.'
generated_text = advanced_nlp.generate_text(prompt)
print(generated_text)

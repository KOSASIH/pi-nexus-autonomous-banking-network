import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class NLPModel(nn.Module):
    def __init__(self, num_classes):
        super(NLPModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.fc(pooled_output)
        return output

class NLPSystem:
    def __init__(self, nlp_model):
        self.nlp_model = nlp_model

    def process_input(self, input_text):
        input_ids = self.nlp_model.tokenizer.encode(input_text, return_tensors='pt')
        attention_mask = self.nlp_model.tokenizer.encode(input_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        output = self.nlp_model(input_ids, attention_mask)
        return output

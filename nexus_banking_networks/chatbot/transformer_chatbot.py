import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

class TransformerChatbot(nn.Module):
    def __init__(self, num_intents, num_slots):
        super(TransformerChatbot, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intents)
        self.slot_tagger = nn.Linear(self.bert_model.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_tagger(pooled_output)
        return intent_logits, slot_logits

# Example usage
chatbot = TransformerChatbot(num_intents=10, num_slots=20)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
intent_logits, slot_logits = chatbot(input_ids, attention_mask)
print(f'Intent logits: {intent_logits}')
print(f'Slot logits: {slot_logits}')

# nlp_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        outputs = self.classifier(pooled_output)
        return outputs

class BertForTokenClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertForTokenClassification, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        outputs = self.classifier(sequence_output)
        return outputs

bert_for_sequence_classification = BertForSequenceClassification(BertModel.from_pretrained('bert-base-uncased'), 8)
bert_for_token_classification = BertForTokenClassification(BertModel.from_pretrained('bert-base-uncased'), 8)

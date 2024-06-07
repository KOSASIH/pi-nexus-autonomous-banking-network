import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoderLayer(input_dim, hidden_dim, num_heads, num_layers)
        self.decoder = TransformerDecoderLayer(hidden_dim, output_dim, num_heads, num_layers)

    def forward(self, input_seq):
        encoder_output = self.encoder(input_seq)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(input_dim, hidden_dim, num_heads)
        self.feed_forward = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_seq):
        # Self-attention mechanism
        attention_output = self.self_attn(input_seq, input_seq)
        # Feed-forward network
        output = self.feed_forward(attention_output)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_heads, num_layers):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, hidden_dim, num_heads)
        self.encoder_attn = MultiHeadAttention(hidden_dim, hidden_dim, num_heads)
        self.feed_forward = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_output):
        # Self-attention mechanism
        attention_output = self.self_attn(encoder_output, encoder_output)
        # Encoder-Decoder attention mechanism
        attention_output = self.encoder_attn(attention_output, encoder_output)
        # Feed-forward network
        output = self.feed_forward(attention_output)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # Compute attention scores
        attention_scores = torch.matmul(query, key.T) / math.sqrt(hidden_dim)
        attention_scores = F.softmax(attention_scores, dim=-1)
        # Compute output
        output = attention_scores * value
        output = self.dropout(output)
        return output

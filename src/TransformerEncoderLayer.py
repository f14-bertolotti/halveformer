from MultiHeadAttention import MultiHeadAttention
import torch

class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, halve=False):
        super(TransformerEncoderLayer, self).__init__()
        self.halve = halve
        
        # Multi-Head Self-Attention Layer
        self.self_attention = MultiHeadAttention(d_model, nhead, dropout=dropout, halve=halve)
        
        # Feed-Forward Network
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer Normalization
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-Attention Block with residual connection
        attn_out = self.dropout(self.self_attention(x, x, x))  # (seq_len, batch_size, d_model)
        if not self.halve:
            x = x + attn_out  # Residual connection
        x = self.layer_norm1(x)  # Layer normalization
        
        # Feed-Forward Network Block with residual connection
        ff_out = self.ffn(x)  # (seq_len, batch_size, d_model)
        x = x + self.dropout(ff_out)  # Residual connection
        x = self.layer_norm2(x)  # Layer normalization
        
        return x


import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1, halve=False):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.halve = halve
        
        # Ensure embed_size is divisible by num_heads
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads."
        
        # Linear layers for query, key, and value
        self.query_linear = torch.nn.Linear(embed_size, embed_size)
        self.key_linear   = torch.nn.Linear(embed_size, embed_size)
        self.value_linear = torch.nn.Linear(embed_size, embed_size)
        
        # Linear layer for the output of attention heads
        self.out_linear = torch.nn.Linear(embed_size, embed_size)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, query, key, value):
        """
        Scaled Dot-Product Attention
        Args:
            query: (batch_size, num_heads, seq_len, head_dim)
            key  : (batch_size, num_heads, seq_len, head_dim)
            value: (batch_size, num_heads, seq_len, head_dim)
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        return attention_output

    def forward(self, query, key, value):
        """
        Forward pass for multi-head attention
        Args:
            query : (batch_size, seq_len, embed_size)
            key   : (batch_size, seq_len, embed_size)
            value : (batch_size, seq_len, embed_size)
        """
        batch_size = query.size(0)
        
        # Linear projections of the input tensors (query, key, value)
        query = self.query_linear(query)  # (batch_size, seq_len, embed_size)
        key   = self.key_linear  (key  )  # (batch_size, seq_len, embed_size)
        value = self.value_linear(value)  # (batch_size, seq_len, embed_size)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key   = key  .view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Apply scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(query, key, value)

        # Concatenate the outputs of all attention heads (batch_size, seq_len, embed_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)

        if self.halve:
            attention_output = attention_output.view(attention_output.size(0), attention_output.size(1)//2, -1, attention_output.size(2)).mean(1)
        
        # Final linear transformation
        output = self.out_linear(attention_output)
        
        return output



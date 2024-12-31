from TransformerEncoderLayer import TransformerEncoderLayer
import transformers
import torch

class HalveTransformer(torch.nn.Module):
    def __init__(
        self, 
        num_classes      = 6   ,
        embedding_size   = 64  ,
        feedforward_size = 128 ,
        dropout          = 0.1 ,
        layers           = 3   ,
        heads            = 2   ,
    ):
        super().__init__()
        
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        self.embedding = torch.nn.Embedding(
            self.bert_tokenizer.vocab_size, 
            embedding_size, 
            padding_idx=self.bert_tokenizer.pad_token_id
        )
        self.encoder = torch.nn.ModuleList([
            TransformerEncoderLayer(
                d_model         = embedding_size   ,
                dim_feedforward = feedforward_size ,
                dropout         = dropout          ,
                nhead           = heads            ,
                halve           = True             
            )
        ] + [
            TransformerEncoderLayer(
                d_model         = embedding_size   ,
                dim_feedforward = feedforward_size ,
                dropout         = dropout          ,
                nhead           = heads            ,
                halve           = False            
            )
            for _ in range(layers-1)
        ])
        self.classifier = torch.nn.Linear(embedding_size, num_classes)

    def forward(self,src):
        src = self.embedding(src).transpose(0,1)
        for layer in self.encoder:
            src = layer(src)
        src = self.classifier(src.mean(0))
        return src

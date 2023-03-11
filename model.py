import torch.nn as nn
import torch.nn.functional as F

# TODO: Check how to inject layers in transformers

class Encoder_Block(nn.Module):
    def __init__(self, 
                adapter,
                d_model = 512, 
                num_heads = 8, 
                dropout = 0.1, 
                hidden_dim = 2048,
    ):
        super(Encoder_Block, self).__init__()
        self.adapter = adapter

        self.mha = nn.MultiheadAttention(d_model, 
                                        num_heads = num_heads,
                                        dropout = dropout,)
        
        self.ff = nn.Sequential(nn.Linear(d_model, hidden_dim),
                                nn.ReLU,
                                nn.Linear(hidden_dim, d_model))
        
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res_x = x.clone()
        x = self.mha(x, x, x)
        x = self.adapter(x)

        res = self.norm_1(res_x + x)

        res_x = res
        x = self.ff(res)
        x = self.adapter(x)

        res = self.norm_2(res_x + x)
        return res

class BERTEncoder(nn.Module):
    def __init__(self,
                n_blocks = 6,
    ):
        super(BERTEncoder, self).__init__()

        self.encoder = nn.ModuleList([Encoder_Block() for _ in range(n_blocks)])

    def forward(self, x):
        for l in self.encoder:
            x = l(x)
        return x

class AdapterModule(nn.Module):
    def __init__(self,
                 in_feature,
                 bottleneck
    ):
        super().__init__()

        self.proj_down = nn.Linear(in_features=in_feature, out_features=bottleneck)
        self.proj_up = nn.Linear(in_features=bottleneck, out_features=in_feature)

    def forward(self, x):
        input = x.clone()

        x = self.proj_down(x)
        x = F.relu(x)
        return self.proj_up(x) + input # Skip Connection

class BERTwithAdapter(nn.Module):
    def __init__(self, model, adapter):
        super().__init__()
        self.embedding = model.embeddings
        self.encoder = BERTEncoder(adapter)
        self.pooler = model.pooler
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.pooler(x)
        x = self.classifier(x)
        return x
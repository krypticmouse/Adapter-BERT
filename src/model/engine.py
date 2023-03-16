import torch.nn as nn
from .bert import BertModel

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x):
        cls_token = self.bert_model(x).last_hidden_state[0,0,:]
        return self.classifier(cls_token)
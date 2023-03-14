import torch
from torch import optim
import torch.nn as nn

from bert import BertModel

model = BertModel.from_pretrained('bert-base-uncased')

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# TODO: Add loggers, parallelism, scheduler
def train(model, epochs, trainloader, valloader):
    for e in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad(set_to_none=True)
            label, sent = batch['label'], batch['sentence']

            output = model(**sent)

            loss = criterion(label, output)
            
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            for batch in valloader:
                label, sent = batch['label'], batch['sentence']

                output = model(**sent)

                loss = criterion(label, output)
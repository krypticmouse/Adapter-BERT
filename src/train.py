import torch
from config import *
from torch import nn
from dataset import CoLADataset

import wandb
from tqdm import tqdm, trange
from model import BertClassifier
from transformers import BertTokenizer, AdamW , get_linear_schedule_with_warmup

class Trainer:
    def __init__(self):
        self.model = BertClassifier(num_labels = 1)
        self.gpu_present = torch.cuda.is_available()
        wandb.init(project="c4ai-adapter-bert")

        if self.gpu_present:
            self.model = self.model.cuda()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_dataset = CoLADataset('data/in_domain_train.tsv', tokenizer)
        self.val_dataset = CoLADataset('data/in_domain_dev.tsv', tokenizer)

        self.loss_fct = nn.BCEWithLogitsLoss()


    def configure_optimizers(self):
        layers = ["adapter", "LayerNorm"]
        params = [p for n, p in self.model.named_parameters() \
                        if any([(nd in n) for nd in layers])]
        
        self.optimizer = AdamW(params, lr=LEARNING_RATE)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps = int(0.1 * len(self.train_dataset))*EPOCHS
        )


    def compute_loss(self, output, labels):
        return self.loss_fct(output, labels)


    def train(self):
        # Setting Up DataLoaders and Optimizars
        trainloader = self.get_train_dataloader()
        valloader = self.get_val_dataloader()
        self.configure_optimizers()

        # Training Loop Starts Here
        for e in trange(EPOCHS):
            train_loss = 0.0
            # Training Step
            for batch in tqdm(trainloader):
                inputs, labels = batch
                if self.gpu_present:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)
                train_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Validation Step
            valid_loss += 0.0
            with torch.no_grad():
                for batch in tqdm(valloader):
                    inputs, labels = batch['sentence'], batch['labels']
                    if self.gpu_present:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    
                    outputs = self.model(**inputs)
                    loss = self.compute_loss(outputs, labels)
                    valid_loss += loss.item()
            
            wandb.log({
                'epoch': e,
                'train_loss': train_loss/len(trainloader),
                'val_loss': valid_loss/len(valloader)
            })
            print(f'Epoch {e}\t\tTraining Loss: {train_loss/len(trainloader)}\t\tValidation Loss: {valid_loss/len(valloader)}')
        wandb.finish()

if __name__=='__main__':
    Trainer().train()
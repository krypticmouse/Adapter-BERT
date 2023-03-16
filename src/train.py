from config import *
from torch import nn
from dataset import CoLADataset

import wandb
from model import BertClassifier
from transformers import Trainer, BertTokenizer, DataCollatorWithPadding, TrainingArguments, AdamW

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("label")
        # forward pass
        outputs = model(**inputs['sentence'])

        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(outputs, labels)
        return (loss, outputs) if return_outputs else loss

def train():
    model = BertClassifier(num_labels = 1)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    layers = ["adapter", "LayerNorm"]
    params = [p for n, p in model.named_parameters() \
                    if any([(nd in n) for nd in layers])]
    optimizer = AdamW(params)

    train_dataset = CoLADataset('data/in_domain_train.tsv', tokenizer)
    val_dataset = CoLADataset('data/in_domain_dev.tsv', tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        "adapter-trainer",
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        evaluation_strategy = 'epoch',
        learning_rate = LEARNING_RATE,
        num_train_epochs = EPOCHS,
        warmup_ratio = 0.1,
        dataloader_num_workers = 2,
        dataloader_drop_last = True,
        seed=42,
        report_to="wandb",
        run_name="c4ai-adapter-bert"
    )

    trainer = CustomTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers = optimizer
    )

    trainer.train()
    wandb.finish()
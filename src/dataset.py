import torch
import pandas as pd
from torch.utils.data import Dataset

class CoLADataset(Dataset):
    def __init__(self, path, tokenizer):
        self.data = pd.read_csv(path, 
                                sep='\t', 
                                names = ["x", "label", "y", "sentence"], 
                                header= None
        )[['label', 'sentence']]
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_pt = self.data.iloc[idx]

        tokenized_sentence = self.tokenizer(data_pt['sentence'], return_tensors= 'pt')
        label = torch.tensor(data_pt["label"])

        return {
            "label": label, 
            "sentence": tokenized_sentence
        }
"""
dataset.py
==========
Handles data loading, preprocessing, and PyTorch Dataset creation.

Usage:
    from dataset import load_pres, SpeechDataset
"""

import codecs
import re
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_pres(fname):
    """
    Load and parse the presidential speech corpus.

    Returns:
        alltxts : list of sentence strings
        alllabs : list of int labels (1=Mitterrand, 0=Chirac)
    """
    alltxts, alllabs = [], []
    s = codecs.open(fname, 'r', 'utf-8')
    while True:
        txt = s.readline()
        if len(txt) < 5:
            break
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*", "\\1", txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", "\\1", txt).strip()
        alllabs.append(1 if "M" in lab else 0)  # Mitterrand=1, Chirac=0
        alltxts.append(txt)
    return alltxts, alllabs


def split_data(alltxts, alllabs, test_size=0.2, random_state=42):
    """
    Stratified train/val split preserving 87/13 class ratio.

    Returns:
        X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        alltxts, alllabs,
        test_size=test_size,
        random_state=random_state,
        stratify=alllabs
    )
    print(f"Train: {len(X_train)} sentences | Val: {len(X_val)} sentences")
    print(f"Train Mitterrand: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
    print(f"Val   Mitterrand: {sum(y_val)}   ({100*sum(y_val)/len(y_val):.1f}%)")
    return X_train, X_val, y_train, y_val


class SpeechDataset(Dataset):
    """
    PyTorch Dataset for tokenized presidential speeches.

    Args:
        texts     : list of raw sentence strings
        labels    : list of int labels (1=Mitterrand, 0=Chirac)
        tokenizer : HuggingFace tokenizer
        max_len   : max token length (default 128)
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }
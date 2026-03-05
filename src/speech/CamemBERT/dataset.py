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
    alltxts, alllabs, alldocids = [], [], []   # add alldocids
    s = codecs.open(fname, 'r', 'utf-8')
    while True:
        txt = s.readline()
        if len(txt) < 5:
            break
        doc_id = re.sub(r"<([0-9]+):[0-9]+:.>.*", "\\1", txt.strip())  # extract "100"
        lab    = re.sub(r"<[0-9]*:[0-9]*:(.)>.*", "\\1", txt)
        txt    = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", "\\1", txt).strip()
        alllabs.append(1 if "M" in lab else 0)
        alltxts.append(txt)
        alldocids.append(doc_id)               # store doc ID
    return alltxts, alllabs, alldocids         # return it


def split_data(alltxts, alllabs, alldocids, test_size=0.2, random_state=42):
    import numpy as np
    from collections import defaultdict

    # Group sentence indices by document
    doc_to_indices = defaultdict(list)
    for i, doc_id in enumerate(alldocids):
        doc_to_indices[doc_id].append(i)

    # Get unique docs and their labels (all sentences in doc share same label)
    unique_docs  = list(doc_to_indices.keys())
    doc_labels   = [alllabs[doc_to_indices[d][0]] for d in unique_docs]

    # Split at DOCUMENT level (stratified by speaker)
    from sklearn.model_selection import train_test_split
    train_docs, val_docs = train_test_split(
        unique_docs,
        test_size=test_size,
        random_state=random_state,
        stratify=doc_labels      # preserve Chirac/Mitterrand ratio at doc level
    )

    train_docs = set(train_docs)
    val_docs   = set(val_docs)

    # Collect sentence indices for each split
    train_idx = [i for i, d in enumerate(alldocids) if d in train_docs]
    val_idx   = [i for i, d in enumerate(alldocids) if d in val_docs]

    X_train  = [alltxts[i]   for i in train_idx]
    X_val    = [alltxts[i]   for i in val_idx]
    y_train  = [alllabs[i]   for i in train_idx]
    y_val    = [alllabs[i]   for i in val_idx]
    ids_train = [alldocids[i] for i in train_idx]
    ids_val   = [alldocids[i] for i in val_idx]

    print(f"Train: {len(X_train)} sentences from {len(train_docs)} documents")
    print(f"Val:   {len(X_val)} sentences from {len(val_docs)} documents")
    print(f"Train Mitterrand: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
    print(f"Val   Mitterrand: {sum(y_val)}   ({100*sum(y_val)/len(y_val):.1f}%)")

    # Verify no document leakage
    assert len(set(ids_train) & set(ids_val)) == 0, "Document leakage detected!"

    return X_train, X_val, y_train, y_val, ids_train, ids_val


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

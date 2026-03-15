import codecs
import re
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold


def load_pres(fname):
    """
    Load and parse the presidential speech corpus.

    Returns:
        alltxts   : list of sentence strings
        alllabs   : list of int labels (1=Mitterrand, 0=Chirac)
        alldocids : list of document IDs (e.g. '100', '101', ...)
    """
    alltxts, alllabs, alldocids = [], [], []
    s = codecs.open(fname, 'r', 'utf-8')
    while True:
        txt = s.readline()
        if len(txt) < 5:
            break
        doc_id = re.sub(r"<([0-9]+):[0-9]+:.>.*", "\\1", txt.strip())
        lab    = re.sub(r"<[0-9]*:[0-9]*:(.)>.*", "\\1", txt)
        txt    = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", "\\1", txt).strip()
        alllabs.append(1 if "M" in lab else 0)
        alltxts.append(txt)
        alldocids.append(doc_id)
    return alltxts, alllabs, alldocids


def split_data(alltxts, alllabs, alldocids=None, test_size=0.2, random_state=42):
    """
    Stratified train/val split preserving 87/13 class ratio.
    Accepts optional alldocids for compatibility.
    """
    if alldocids is not None:
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            alltxts, alllabs, alldocids,
            test_size=test_size,
            random_state=random_state,
            stratify=alllabs
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            alltxts, alllabs,
            test_size=test_size,
            random_state=random_state,
            stratify=alllabs
        )
        ids_train, ids_val = None, None

    print(f"Train: {len(X_train)} sentences | Val: {len(X_val)} sentences")
    print(f"Train Mitterrand: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
    print(f"Val   Mitterrand: {sum(y_val)}   ({100*sum(y_val)/len(y_val):.1f}%)")

    if alldocids is not None:
        return X_train, X_val, y_train, y_val, ids_train, ids_val
    return X_train, X_val, y_train, y_val


def kfold_splits(alltxts, alllabs, n_splits=5, random_state=42):
    """
    Yields (X_train, X_val, y_train, y_val) for each fold.
    Uses StratifiedKFold to preserve class ratio in every fold.

    Args:
        alltxts     : list of all sentences
        alllabs     : list of all labels
        n_splits    : number of folds (default 5)
        random_state: for reproducibility

    Usage:
        for fold, (X_train, X_val, y_train, y_val) in enumerate(kfold_splits(alltxts, alllabs)):
            ...
    """
    import numpy as np
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X   = list(alltxts)
    y   = list(alllabs)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = [X[i] for i in train_idx]
        X_val   = [X[i] for i in val_idx]
        y_train = [y[i] for i in train_idx]
        y_val   = [y[i] for i in val_idx]

        print(f"\nFold {fold+1}/{n_splits}")
        print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
        print(f"  Train Mitterrand: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
        print(f"  Val   Mitterrand: {sum(y_val)}   ({100*sum(y_val)/len(y_val):.1f}%)")

        yield fold, X_train, X_val, y_train, y_val




def augment_mitterrand(X_train, y_train, multiplier=3, random_state=42):
    """
    Augments Mitterrand sentences in training set by:
      1. Simple duplication (always applied)
      2. Random word deletion (drops ~10% of words randomly)
      3. Random word swap (swaps two adjacent words)

    Args:
        X_train    : list of training sentences
        y_train    : list of training labels
        multiplier : how many augmented copies per Mitterrand sentence (default 3)

    Returns:
        X_aug, y_aug : augmented training set (original + augmented Mitterrand)
    """
    random.seed(random_state)

    mitterrand_texts = [t for t, l in zip(X_train, y_train) if l == 1]
    print(f"Augmenting {len(mitterrand_texts)} Mitterrand sentences (x{multiplier})...")

    augmented_texts  = []
    augmented_labels = []

    for text in mitterrand_texts:
        words = text.split()
        if not words:
            continue

        for i in range(multiplier):
            if i == 0:
                # Copy 1: simple duplication
                aug = text

            elif i == 1:
                # Copy 2: random word deletion (~10% of words dropped)
                aug_words = [w for w in words if random.random() > 0.1]
                aug = " ".join(aug_words) if aug_words else text

            else:
                # Copy 3+: random adjacent word swap
                aug_words = words.copy()
                if len(aug_words) > 1:
                    idx = random.randint(0, len(aug_words) - 2)
                    aug_words[idx], aug_words[idx+1] = aug_words[idx+1], aug_words[idx]
                aug = " ".join(aug_words)

            augmented_texts.append(aug)
            augmented_labels.append(1)

    X_aug = X_train + augmented_texts
    y_aug = y_train + augmented_labels

    print(f"After augmentation:")
    print(f"  Total:      {len(X_aug)} sentences")
    print(f"  Chirac:     {sum(1 for y in y_aug if y == 0)}")
    print(f"  Mitterrand: {sum(1 for y in y_aug if y == 1)}")

    return X_aug, y_aug


def add_context(texts, doc_ids, window=2, sep_token="</s>"):
    """
    Adds context only from the same document.
    window=2 means 2 sentences before and after.
    """
    contextualized = []
    for i, text in enumerate(texts):
        left, right = [], []

        for j in range(max(0, i - window), i):
            if doc_ids[j] == doc_ids[i]:
                left.append(texts[j])

        for j in range(i + 1, min(len(texts), i + window + 1)):
            if doc_ids[j] == doc_ids[i]:
                right.append(texts[j])

        parts = left + [text] + right
        contextualized.append(f" {sep_token} ".join(parts))

    return contextualized


class SpeechDataset(Dataset):
    """
    PyTorch Dataset for tokenized presidential speeches.

    Args:
        texts     : list of raw sentence strings
        labels    : list of int labels (1=Mitterrand, 0=Chirac)
        tokenizer : HuggingFace tokenizer
        max_len   : max token length (default 256)
    """
    def __init__(self, texts, labels, tokenizer, max_len=256):
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
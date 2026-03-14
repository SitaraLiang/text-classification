"""
evaluate.py
===========
Loads a saved checkpoint and evaluates on validation set, or generates
submission probabilities for the test set.

Usage:
    # Evaluate on validation set
    python evaluate.py --fname ../../data/corpus.tache1.learn.utf8 --checkpoint ./checkpoints/best_model

    # Generate submission file from test set
    python evaluate.py --checkpoint ./checkpoints/best_model --test_fname ../../data/test.utf8 --submission submission.txt
"""

import argparse
import numpy as np
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, classification_report
)

from dataset import load_pres, split_data, SpeechDataset


def load_checkpoint(checkpoint_path):
    """
    Loads tokenizer and model from a saved checkpoint directory.
    The checkpoint must have been saved with tokenizer.save_pretrained().
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    tokenizer = CamembertTokenizer.from_pretrained(checkpoint_path)
    model     = CamembertForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    print("Checkpoint loaded.")
    return tokenizer, model


def predict_probs(model, tokenizer, texts, batch_size=32):
    """
    Returns P(Mitterrand) for each sentence in texts.
    Runs inference in batches to avoid OOM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} sentences...")

    return np.array(all_probs)


def evaluate(checkpoint_path, fname):
    tokenizer, model = load_checkpoint(checkpoint_path)

    # Reload the same split used during training
    alltxts, alllabs, _ = load_pres(fname)

    from collections import Counter
    counts = Counter(alllabs)
    print(f"Label distribution: {counts}")
    print(f"  0 (Chirac):     {counts[0]}")
    print(f"  1 (Mitterrand): {counts[1]}")

    _, X_val, _, y_val = split_data(alltxts, alllabs)

    print("\nRunning inference on validation set...")
    probs = predict_probs(model, tokenizer, X_val)
    preds = (probs >= 0.5).astype(int)

    f1  = f1_score(y_val, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_val, probs)
    ap  = average_precision_score(y_val, probs, pos_label=1)

    print(f"\n{'─'*40}")
    print(f"  CamemBERT — Checkpoint Evaluation")
    print(f"  {checkpoint_path}")
    print(f"{'─'*40}")
    print(f"  F1  (Mitterrand): {f1:.4f}")
    print(f"  AUC (ROC):        {auc:.4f}")
    print(f"  AP  (PR curve):   {ap:.4f}")
    print(f"{'─'*40}")
    print(classification_report(y_val, preds, target_names=["Chirac", "Mitterrand"]))

    return probs, y_val, X_val


def generate_submission(checkpoint_path, test_fname, output_path):
    """
    Loads test sentences (no labels), predicts P(Mitterrand), saves to file.
    Test file format: same as train but labels can be anything (ignored).
    """
    tokenizer, model = load_checkpoint(checkpoint_path)

    # Load test sentences — strip labels if present, otherwise load raw lines
    test_texts = []
    with open(test_fname, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) < 2:
                continue
            # Strip label tag if present, otherwise use line as-is
            import re
            txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", "\\1", line).strip()
            if not txt:
                txt = line.strip()
            test_texts.append(txt)

    print(f"\nLoaded {len(test_texts)} test sentences.")
    print("Running inference...")
    probs = predict_probs(model, tokenizer, test_texts)

    with open(output_path, 'w') as f:
        for p in probs:
            f.write(f"{p:.6f}\n")

    print(f"\nSubmission saved → {output_path} ({len(probs)} lines)")
    print(f"  Predicted Mitterrand (p>0.5): {sum(probs > 0.5)}")
    print(f"  Predicted Chirac     (p<0.5): {sum(probs < 0.5)}")
    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CamemBERT speaker classifier")
    parser.add_argument("--checkpoint",  type=str, required=True,  help="Path to saved checkpoint (best_model dir)")
    parser.add_argument("--fname",       type=str, default=None,   help="Training corpus path (for val evaluation)")
    parser.add_argument("--test_fname",  type=str, default=None,   help="Test corpus path (for submission generation)")
    parser.add_argument("--submission",  type=str, default="submission_camembert.txt", help="Output submission file path")
    args = parser.parse_args()

    if args.test_fname:
        # Generate submission for test set
        generate_submission(args.checkpoint, args.test_fname, args.submission)

    elif args.fname:
        # Evaluate on validation set
        probs, y_val, X_val = evaluate(args.checkpoint, args.fname)
        wrong_mitterrand = [
            (text, prob)
            for text, label, prob in zip(X_val, y_val, probs)
            if label == 1 and prob < 0.3
        ]

        easy_mitterrand = [
            (text, prob)
            for text, label, prob in zip(X_val, y_val, probs)
            if label == 1 and prob > 0.7
        ]

        print(f"Strongly MISSED Mitterrand: {len(wrong_mitterrand)}")
        print(f"Correctly found Mitterrand: {len(easy_mitterrand)}")
        print(f"\n--- 20 most missed Mitterrand sentences ---")
        for text, prob in sorted(wrong_mitterrand, key=lambda x: x[1])[:20]:
            print(f"  p={prob:.3f} | {text.strip()[:100]}")

        print(f"\n--- 20 easiest Mitterrand sentences ---")
        for text, prob in sorted(easy_mitterrand, key=lambda x: x[1], reverse=True)[:20]:
            print(f"  p={prob:.3f} | {text.strip()[:100]}")

    else:
        print("Please provide either --fname (for validation) or --test_fname (for submission).")
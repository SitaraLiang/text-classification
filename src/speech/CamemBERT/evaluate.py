"""
evaluate.py
===========
Loads a saved checkpoint and evaluates on validation set, or generates
submission probabilities for the test set.

IMPORTANT: Always pass --context if the model was trained with --context.

Usage:
    # Evaluate on validation set (with context)
    python evaluate.py --checkpoint ./checkpoints/best_model --fname ../../data/corpus.tache1.learn.utf8 --context

    # Generate submission file from test set (with context)
    python evaluate.py --checkpoint ./checkpoints/best_model --test_fname ../../data/test.utf8 --submission submission.txt --context

    # Without context (only if model was trained without --context)
    python evaluate.py --checkpoint ./checkpoints/best_model --fname ../../data/corpus.tache1.learn.utf8
"""

import argparse
import re
import numpy as np
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, classification_report
)

from dataset import load_pres, split_data, add_context


def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    tokenizer = CamembertTokenizer.from_pretrained(checkpoint_path)
    model     = CamembertForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    print("Checkpoint loaded.")
    return tokenizer, model

def predict_probs(model, tokenizer, texts, batch_size=32):
    """
    Returns P(Mitterrand) for each sentence.
    Runs inference in batches to avoid OOM.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
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


def evaluate(checkpoint_path, fname, use_context=False):
    tokenizer, model = load_checkpoint(checkpoint_path)

    alltxts, alllabs, alldocids = load_pres(fname)

    # ── Apply context BEFORE split (mirrors train.py exactly) ──
    if use_context:
        print("Adding sentence context (window=2)...")
        alltxts = add_context(alltxts, alldocids, window=2)

    # ── Split after context (same random_state=42 as train.py) ──
    _, X_val, _, y_val = split_data(alltxts, alllabs)

    print("\nRunning inference on validation set...")
    probs = predict_probs(model, tokenizer, X_val)
    preds = (probs >= 0.5).astype(int)

    f1  = f1_score(y_val, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_val, probs)
    ap  = average_precision_score(y_val, probs, pos_label=1)

    print(f"\n{'─'*40}")
    print(f"  CamemBERT — Checkpoint Evaluation")
    print(f"  context={'yes (window=2)' if use_context else 'no'}")
    print(f"{'─'*40}")
    print(f"  F1  (Mitterrand): {f1:.4f}")
    print(f"  AUC (ROC):        {auc:.4f}")
    print(f"  AP  (PR curve):   {ap:.4f}")
    print(f"{'─'*40}")
    print(classification_report(y_val, preds, target_names=["Chirac", "Mitterrand"]))

    # ── Threshold tuning ──
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores  = [f1_score(y_val, (probs >= t).astype(int), pos_label=1)
                  for t in thresholds]
    best_t = thresholds[np.argmax(f1_scores)]
    print(f"\n  Best threshold: {best_t:.2f} → F1={max(f1_scores):.4f}")

    return probs, y_val, X_val


def generate_submission(checkpoint_path, test_fname, output_path, use_context=False):
    """
    One probability per line — prof applies 0.5 cutoff on their end.
    IMPORTANT: use_context must match what was used during training.
    """
    tokenizer, model = load_checkpoint(checkpoint_path)

    test_texts, test_docids = [], []
    with open(test_fname, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) < 2:
                continue
            doc_id = re.sub(r"<([0-9]+):[0-9]+:.>.*", "\\1", line.strip())
            txt    = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", "\\1", line).strip()
            if not txt:
                txt = line.strip()
                doc_id = str(len(test_texts))  # fallback doc_id for untagged lines
            test_texts.append(txt)
            test_docids.append(doc_id)

    print(f"\nLoaded {len(test_texts)} test sentences.")

    # ── Apply same context window used during training ──
    if use_context:
        print("Adding sentence context (window=2)...")
        test_texts = add_context(test_texts, test_docids, window=2)
        print(f"Context added.")

    print("Running inference...")
    probs = predict_probs(model, tokenizer, test_texts)

    with open(output_path, 'w') as f:
        for p in probs:
            f.write(f"{p:.6f}\n")

    print(f"\nSubmission saved → {output_path} ({len(probs)} lines)")
    print(f"  Predicted Mitterrand (p>0.5): {sum(probs > 0.5)}")
    print(f"  Predicted Chirac     (p<0.5): {sum(probs < 0.5)}")
    return probs


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CamemBERT speaker classifier")
    parser.add_argument("--checkpoint", type=str, required=True,                      help="Path to saved checkpoint (best_model dir)")
    parser.add_argument("--fname",      type=str, default=None,                       help="Training corpus path (for val evaluation)")
    parser.add_argument("--test_fname", type=str, default=None,                       help="Test corpus path (for submission generation)")
    parser.add_argument("--submission", type=str, default="submission_camembert.txt", help="Output submission file path")
    parser.add_argument("--context",    action="store_true",                          help="Apply context window=2 (must match training config)")
    args = parser.parse_args()

    if args.test_fname:
        generate_submission(
            args.checkpoint, args.test_fname,
            args.submission, use_context=args.context
        )
    elif args.fname:
        probs, y_val, X_val = evaluate(
            args.checkpoint, args.fname,
            use_context=args.context
        )

        """
        # ── Diagnosis: missed vs easy Mitterrand sentences ──
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
        print(f"\nStrongly MISSED Mitterrand: {len(wrong_mitterrand)}")
        print(f"Correctly found Mitterrand: {len(easy_mitterrand)}")
        print(f"\n--- 20 most missed Mitterrand sentences ---")
        for text, prob in sorted(wrong_mitterrand, key=lambda x: x[1])[:20]:
            print(f"  p={prob:.3f} | {text.strip()[:100]}")
        print(f"\n--- 20 easiest Mitterrand sentences ---")
        for text, prob in sorted(easy_mitterrand, key=lambda x: x[1], reverse=True)[:20]:
            print(f"  p={prob:.3f} | {text.strip()[:100]}")
        """
    else:
        print("Please provide either --fname (validation) or --test_fname (submission).")
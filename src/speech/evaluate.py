import argparse
import re
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, classification_report
)

from speech.dataset import load_pres, split_data, add_context

def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    tokenizer = CamembertTokenizer.from_pretrained(checkpoint_path)
    model     = CamembertForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    print("Checkpoint loaded.")
    return tokenizer, model

def predict_probs(model, tokenizer, texts, temperature=1.0, batch_size=32):
    """
    Returns P(Chirac) for each sentence.
    temperature > 1.0 spreads out overconfident probabilities.
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
            logits = logits / temperature 
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(Chirac)

        all_probs.extend(probs)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} sentences...")

    return np.array(all_probs)


def find_temperature(model, tokenizer, X_val, y_val):
    """
    Tries different temperatures and prints metrics.
    AUC should stay stable — AP and F1 may improve.
    """
    print(f"\n{'─'*50}")
    print(f"  Temperature search")
    print(f"{'─'*50}")
    print(f"  {'T':>6} | {'min':>6} | {'max':>6} | {'std':>6} | {'AUC':>6} | {'AP':>6} | {'F1':>6}")
    print(f"  {'─'*55}")

    best_ap, best_t = 0, 1.0
    for T in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        probs = predict_probs(model, tokenizer, X_val, temperature=T)
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_val, probs)
        ap  = average_precision_score(y_val, probs, pos_label=1)
        f1  = f1_score(y_val, preds, pos_label=1, zero_division=0)
        print(f"  {T:.16f} | {probs.min()::.16f} | {probs.max():.16f} | "
              f"{probs.std():.16f} | {auc:.16f} | {ap:.16f} | {f1:.16f}")
        if ap > best_ap:
            best_ap, best_t = ap, T

    print(f"\n  Best temperature by AP: T={best_t} → AP={best_ap:.16f}")
    return best_t

def evaluate(checkpoint_path, fname, use_context=False, fold=None, n_folds=5,
             temperature=1.0, find_temp=False):
    tokenizer, model = load_checkpoint(checkpoint_path)

    alltxts, alllabs, alldocids = load_pres(fname)

    if use_context:
        print("Adding sentence context (window=2)...")
        alltxts = add_context(alltxts, alldocids, window=2)

    if fold is not None:
        skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(skf.split(alltxts, alllabs))
        _, val_idx = splits[fold - 1]
        X_val  = [alltxts[i] for i in val_idx]
        y_val  = [alllabs[i] for i in val_idx]
        print(f"Fold {fold}/{n_folds} val set: {len(X_val)} sentences")
    else:
        _, X_val, _, y_val = split_data(alltxts, alllabs)

    # ── Temperature search ──
    if find_temp:
        temperature = find_temperature(model, tokenizer, X_val, y_val)

    print(f"\nRunning inference (temperature={temperature})...")
    probs = predict_probs(model, tokenizer, X_val, temperature=temperature)
    preds = (probs > 0.5).astype(int)

    f1  = f1_score(y_val, preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(y_val, probs)
    ap  = average_precision_score(y_val, probs, pos_label=1)

    print(f"\n{'─'*40}")
    print(f"  CamemBERT — Checkpoint Evaluation")
    print(f"  context    : {'yes (window=2)' if use_context else 'no'}")
    print(f"  fold       : {fold if fold else 'standard 80/20'}")
    print(f"  temperature: {temperature}")
    print(f"{'─'*40}")
    print(f"  F1  (Mitterrand): {f1:.4f}")
    print(f"  AUC (ROC):        {auc:.4f}")
    print(f"  AP  (PR curve):   {ap:.4f}")
    print(f"{'─'*40}")
    print(classification_report(y_val, preds, target_names=["Chirac", "Mitterrand"]))

    return probs, y_val, X_val, temperature

def viterbi_smooth(probs, doc_ids,
                   p_mm=0.947,  # P(Mitterrand|Mitterrand)
                   p_cc=0.992,  # P(Chirac|Chirac)
                   p_prior_m=0.13):
    """
    Apply Viterbi decoding within each document separately.
    probs_m : P(Chirac) from CamemBERT for each sentence
    doc_ids : document ID for each sentence (to respect boundaries)
    """
    p_prior_c = 1 - p_prior_m
    
    # Transition matrix T[from][to]
    T = np.array([[p_cc,     1-p_cc],   # from Chirac
                  [1-p_mm,   p_mm  ]])  # from Mitterrand

    smoothed = np.zeros(len(probs))
    
    # Group by document to respect boundaries
    from itertools import groupby
    from operator import itemgetter
    
    # Get unique doc segments with indices
    doc_segments = []
    for doc_id, group in groupby(enumerate(doc_ids), key=lambda x: x[1]):
        indices = [i for i, _ in group]
        doc_segments.append(indices)
    
    for indices in doc_segments:
        n = len(indices)
        probs = probs[indices]  # P(Mitterrand) for this doc
        
        # Emission probabilities
        # emit[t][s] = P(observation at t | state s)
        emit = np.array([[1-p, p] for p in probs])  # shape (n, 2)
        
        # Viterbi
        viterbi = np.zeros((n, 2))
        backptr = np.zeros((n, 2), dtype=int)
        
        # Initialization
        viterbi[0] = [p_prior_c * emit[0][0],
                      p_prior_m * emit[0][1]]
        viterbi[0] /= viterbi[0].sum()  # normalize
        
        # Forward pass
        for t in range(1, n):
            for s in range(2):
                scores = viterbi[t-1] * T[:, s] * emit[t][s]
                backptr[t][s] = np.argmax(scores)
                viterbi[t][s] = np.max(scores)
            viterbi[t] /= viterbi[t].sum()  # normalize
        
        # Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n-2, -1, -1):
            states[t] = backptr[t+1][states[t+1]]
        
        for i, idx in enumerate(indices):
            if states[i] == 0:  # Chirac
                smoothed[idx] = 0.5 + (probs[idx] * 0.5)   # push above 0.5
            else:               # Mitterrand
                smoothed[idx] = 0.5 - ((1-probs[idx]) * 0.5)  # push below 0.5
                    
    return smoothed


def generate_submission(checkpoint_path, test_fname, output_path,
                        use_context=False, temperature=1.0):
    tokenizer, model = load_checkpoint(checkpoint_path)

    test_texts, test_docids = [], []
    with open(test_fname, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) < 2:
                continue
            doc_id = re.sub(r"<([0-9]+):[0-9]+>.*", "\\1", line.strip())
            txt    = re.sub(r"<[0-9]+:[0-9]+>(.*)", "\\1", line).strip()
            if not txt:
                txt    = line.strip()
                doc_id = str(len(test_texts))
            test_texts.append(txt)
            test_docids.append(doc_id)

    print(f"Loaded {len(test_texts)} sentences | Unique docs: {len(set(test_docids))}")

    if use_context:
        print("Adding sentence context (window=2)...")
        test_texts = add_context(test_texts, test_docids, window=2)

    print(f"Running inference (temperature={temperature})...")
    probs = predict_probs(model, tokenizer, test_texts, temperature=temperature)
    smoothed_probs = viterbi_smooth(probs, test_docids)
    

    with open(output_path, 'w') as f:
        for p in smoothed_probs:
            f.write(f"{p:.16f}\n")

    print(f"\nSubmission saved to {output_path} ({len(smoothed_probs)} lines)")
    print(f"  Chirac     (p>0.5): {sum(smoothed_probs > 0.5)}")
    print(f"  Mitterrand (p<0.5): {sum(smoothed_probs < 0.5)}")
    return smoothed_probs


def generate_submission_ensemble(fold_checkpoints, test_fname, output_path,
                                 use_context=False, temperature=1.0):
    test_texts, test_docids = [], []
    with open(test_fname, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) < 2:
                continue
            doc_id = re.sub(r"<([0-9]+):[0-9]+>.*", "\\1", line.strip())
            txt    = re.sub(r"<[0-9]+:[0-9]+>(.*)", "\\1", line).strip()
            if not txt:
                txt    = line.strip()
                doc_id = str(len(test_texts))
            test_texts.append(txt)
            test_docids.append(doc_id)

    print(f"Loaded {len(test_texts)} sentences | Unique docs: {len(set(test_docids))}")

    if use_context:
        print("Adding sentence context (window=2)...")
        test_texts = add_context(test_texts, test_docids, window=2)

    # Collect P(Chirac) from each fold
    all_probs = []
    for i, checkpoint_path in enumerate(fold_checkpoints):
        print(f"\nFold {i+1}/{len(fold_checkpoints)}: loading...")
        tokenizer, model = load_checkpoint(checkpoint_path)
        probs = predict_probs(model, tokenizer, test_texts, temperature=temperature)
        all_probs.append(probs)
        print(f"  Chirac (p>0.5): {sum(probs > 0.5)}")

    ensemble = np.mean(all_probs, axis=0)
    smoothed_probs = viterbi_smooth(ensemble, test_docids)

    print(f"\n{'─'*40}")
    print(f"  Ensemble of {len(fold_checkpoints)} folds (T={temperature})")
    print(f"  Chirac     (p>0.5): {sum(smoothed_probs > 0.5)}")
    print(f"  Mitterrand (p<0.5): {sum(smoothed_probs < 0.5)}")
    print(f"{'─'*40}")

    with open(output_path, 'w') as f:
        for p in smoothed_probs:
            f.write(f"{p:.16f}\n")

    print(f"Ensemble submission saved to {output_path} ({len(smoothed_probs)} lines)")
    return smoothed_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CamemBERT speaker classifier")
    parser.add_argument("--checkpoint", type=str, default=None,                       help="Path to single checkpoint (for non-ensemble)")
    parser.add_argument("--ckpt_dir",   type=str, default=None,                       help="Base dir containing fold_1..fold_5 (for ensemble)")
    parser.add_argument("--fname",      type=str, default=None,                       help="Training corpus path (for val evaluation)")
    parser.add_argument("--test_fname", type=str, default=None,                       help="Test corpus path (for submission)")
    parser.add_argument("--submission", type=str, default="submission-pres-1.csv",    help="Output submission file path")
    parser.add_argument("--context",    action="store_true",                          help="Apply context window=2")
    parser.add_argument("--fold",       type=int, default=None,                       help="Which fold val set to evaluate on (1-5)")
    parser.add_argument("--n_folds",    type=int, default=5,                          help="Total folds used during training")
    parser.add_argument("--ensemble",   action="store_true",                          help="Ensemble all folds for submission")
    parser.add_argument("--temperature",type=float, default=1.0,                      help="Temperature for calibration (default 1.0 = no scaling)")
    parser.add_argument("--find_temp",  action="store_true",                          help="Search for best temperature on val set")
    args = parser.parse_args()

    if args.ensemble and args.test_fname:
        # ── Ensemble submission ──
        fold_checkpoints = [f"{args.ckpt_dir}/fold_{i}/best_model" for i in range(1, 6)]
        generate_submission_ensemble(
            fold_checkpoints=fold_checkpoints,
            test_fname=args.test_fname,
            output_path=args.submission,
            use_context=args.context,
            temperature=args.temperature
        )

    elif args.test_fname and args.checkpoint:
        # ── Single checkpoint submission ──
        generate_submission(
            checkpoint_path=args.checkpoint,
            test_fname=args.test_fname,
            output_path=args.submission,
            use_context=args.context,
            temperature=args.temperature
        )

    elif args.fname:
        # ── Validation evaluation ──
        checkpoint = args.checkpoint or f"{args.ckpt_dir}/fold_{args.fold}/best_model"
        probs, y_val, X_val, best_t = evaluate(
            checkpoint_path=checkpoint,
            fname=args.fname,
            use_context=args.context,
            fold=args.fold,
            n_folds=args.n_folds,
            temperature=args.temperature,
            find_temp=args.find_temp
        )
    else:
        print("Please provide --fname (validation) or --test_fname (submission).")
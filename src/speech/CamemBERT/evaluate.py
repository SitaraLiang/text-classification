import argparse
import re
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
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

def predict_probs(model, tokenizer, texts, temperature=1.0, batch_size=32):
    """
    Returns P(Mitterrand) for each sentence.
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
            logits = logits / temperature          # ← temperature scaling
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(Mitterrand)

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
        print(f"  {T:>6.1f} | {probs.min():>6.3f} | {probs.max():>6.3f} | "
              f"{probs.std():>6.3f} | {auc:>6.4f} | {ap:>6.4f} | {f1:>6.4f}")
        if ap > best_ap:
            best_ap, best_t = ap, T

    print(f"\n  Best temperature by AP: T={best_t} → AP={best_ap:.4f}")
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
    preds = (probs >= 0.5).astype(int)

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
    probs_m = predict_probs(model, tokenizer, test_texts, temperature=temperature)
    probs_c = 1 - probs_m     # ← P(Chirac) for submission

    with open(output_path, 'w') as f:
        for p in probs_c:
            f.write(f"{p:.6f}\n")

    print(f"\nSubmission saved → {output_path} ({len(probs_c)} lines)")
    print(f"  Chirac     (p>0.5): {sum(probs_c > 0.5)}")
    print(f"  Mitterrand (p<0.5): {sum(probs_c < 0.5)}")
    return probs_c


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

    # Collect P(Mitterrand) from each fold
    all_probs = []
    for i, checkpoint_path in enumerate(fold_checkpoints):
        print(f"\nFold {i+1}/{len(fold_checkpoints)}: loading...")
        tokenizer, model = load_checkpoint(checkpoint_path)
        probs = predict_probs(model, tokenizer, test_texts, temperature=temperature)
        all_probs.append(probs)
        print(f"  Mitterrand (p>0.5): {sum(probs > 0.5)}")

    # Average P(Mitterrand) across folds, then flip to P(Chirac)
    ensemble_m = np.mean(all_probs, axis=0)
    ensemble_c = 1 - ensemble_m    # P(Chirac) for submission

    print(f"\n{'─'*40}")
    print(f"  Ensemble of {len(fold_checkpoints)} folds (T={temperature})")
    print(f"  Chirac     (p>0.5): {sum(ensemble_c > 0.5)}")
    print(f"  Mitterrand (p<0.5): {sum(ensemble_c < 0.5)}")
    print(f"{'─'*40}")

    with open(output_path, 'w') as f:
        for p in ensemble_c:
            f.write(f"{p:.6f}\n")

    print(f"Ensemble submission saved → {output_path} ({len(ensemble_c)} lines)")
    return ensemble_c


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
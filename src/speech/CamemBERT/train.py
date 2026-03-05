"""
train.py
========
Trains CamemBERT for speaker classification (Chirac / Mitterrand).
Saves best checkpoint to --output_dir.

Usage:
    python train.py --fname ../../data/corpus.tache1.learn.utf8
    python train.py --fname ../../data/corpus.tache1.learn.utf8 --output_dir ./checkpoints --strategy full --epochs 5
"""

import argparse
import torch
import torch.nn as nn
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from dataset import load_pres, split_data, SpeechDataset


# ─────────────────────────────────────────
# Weighted Trainer (handles class imbalance)
# ─────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────
# Metrics (called after each eval epoch)
# ─────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "f1":  round(f1_score(labels, preds, pos_label=1, zero_division=0), 4),
        "auc": round(roc_auc_score(labels, probs), 4),
        "ap":  round(average_precision_score(labels, probs, pos_label=1), 4),
    }


# ─────────────────────────────────────────
# Freeze strategy
# ─────────────────────────────────────────
def freeze_strategy(model, strategy="top_layers"):
    encoder_attr = next(
        (name for name in ["roberta", "camembert", "bert"] if hasattr(model, name)),
        None
    )
    encoder = getattr(model, encoder_attr)
    print(f"Detected encoder attribute: '{encoder_attr}'")

    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif strategy == "head_only":
        for param in encoder.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif strategy == "top_layers":
        for param in encoder.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(encoder.encoder.layer):
            for param in layer.parameters():
                param.requires_grad = i >= 10  # train only layers 10-11
        for param in model.classifier.parameters():
            param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Strategy : '{strategy}'")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model


# ─────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────
def train(fname, output_dir, strategy, epochs, batch_size):

    # Load & split data
    alltxts, alllabs, alldocids = load_pres(fname)
    X_train, X_val, y_train, y_val, ids_train, ids_val = split_data(alltxts, alllabs, alldocids)
    # ids_train and ids_val are not used during training, but needed for evaluate.py

    # Tokenizer & model
    print("\nLoading CamemBERT...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model     = CamembertForSequenceClassification.from_pretrained(
        "camembert-base", num_labels=2
    )
    model = freeze_strategy(model, strategy=strategy)

    # Class weights
    n_total      = len(y_train)
    n_chirac     = sum(1 for y in y_train if y == 0)
    n_mitterrand = sum(1 for y in y_train if y == 1)
    w_chirac     = n_total / (2 * n_chirac)
    w_mitterrand = n_total / (2 * n_mitterrand)
    class_weights = torch.tensor([w_chirac, w_mitterrand], dtype=torch.float)

    print(f"Class weights → Chirac: {w_chirac:.3f} | Mitterrand: {w_mitterrand:.3f}")

    # Datasets
    train_dataset = SpeechDataset(X_train, y_train, tokenizer)
    val_dataset   = SpeechDataset(X_val,   y_val,   tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.1,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="ap",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    # Save best model + tokenizer together
    best_path = f"{output_dir}/best_model"
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)
    print(f"\nBest model saved → {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CamemBERT speaker classifier")
    parser.add_argument("--fname",      type=str, required=True,             help="Path to training corpus")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",   help="Where to save checkpoints")
    parser.add_argument("--strategy",   type=str, default="top_layers",      help="Freeze strategy: full | head_only | top_layers")
    parser.add_argument("--epochs",     type=int, default=3,                 help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,                help="Per-device train batch size")
    args = parser.parse_args()

    train(args.fname, args.output_dir, args.strategy, args.epochs, args.batch_size)

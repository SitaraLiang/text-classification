import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from dataset import load_pres, split_data, kfold_splits, augment_mitterrand, add_context, SpeechDataset


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "f1":  round(f1_score(labels, preds, pos_label=1, zero_division=0), 4),
        "auc": round(roc_auc_score(labels, probs), 4),
        "ap":  round(average_precision_score(labels, probs, pos_label=1), 4),
    }


def freeze_strategy(model, strategy="full"):
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
                param.requires_grad = i >= 8
        for param in model.classifier.parameters():
            param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Strategy : '{strategy}'")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model

def get_class_weights(y_train):
    n_total      = len(y_train)
    n_chirac     = sum(1 for y in y_train if y == 0)
    n_mitterrand = sum(1 for y in y_train if y == 1)
    w_chirac     = n_total / (2 * n_chirac)
    w_mitterrand = n_total / (2 * n_mitterrand)
    print(f"Class weights to Chirac: {w_chirac:.3f} | Mitterrand: {w_mitterrand:.3f}")
    return torch.tensor([w_chirac, w_mitterrand], dtype=torch.float)


def train_one(X_train, y_train, X_val, y_val,
              tokenizer, output_dir, epochs, batch_size, lr, augment, strategy):

    model = CamembertForSequenceClassification.from_pretrained(
        "camembert-base", num_labels=2
    )
    model = freeze_strategy(model, strategy=strategy) 

    if augment:
        X_train, y_train = augment_mitterrand(X_train, y_train, multiplier=3)

    class_weights = get_class_weights(y_train)

    train_dataset = SpeechDataset(X_train, y_train, tokenizer, max_len=256)
    val_dataset   = SpeechDataset(X_val,   y_val,   tokenizer, max_len=256)

    # Calculate warmup_steps explicitly (replaces deprecated warmup_ratio)
    steps_per_epoch = len(X_train) // batch_size
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = int(0.1 * total_steps)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=warmup_steps,
        learning_rate=lr,
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

    trainer.train()
    return trainer

def train(args):
    alltxts, alllabs, alldocids = load_pres(args.fname)

    # Add context BEFORE any split or data augmentation (preserves real speech order)
    if args.context:
        print("Adding sentence context (window=1)...")
        alltxts = add_context(alltxts, alldocids, window=2)
        print(f"Context added to {len(alltxts)} sentences.")


    print("\nLoading CamemBERT tokenizer...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # ── K-Fold mode ──
    if args.kfold > 1:
        print(f"\n{'='*50}")
        print(f"K-Fold Cross-Validation ({args.kfold} folds)")
        print(f"{'='*50}")

        fold_scores = []

        for fold, X_train, X_val, y_train, y_val in kfold_splits(
            alltxts, alllabs, n_splits=args.kfold
        ):
            fold_dir = f"{args.output_dir}/fold_{fold+1}"
            print(f"\n{'─'*40}")
            print(f"Training fold {fold+1}/{args.kfold}...")
            print(f"{'─'*40}")

            trainer = train_one(
                X_train, y_train, X_val, y_val,
                tokenizer, fold_dir,
                args.epochs, args.batch_size, args.lr, args.augment, args.strategy 
            )

            # Evaluate this fold
            val_dataset = SpeechDataset(X_val, y_val, tokenizer, max_len=256)
            preds_out   = trainer.predict(val_dataset)
            logits      = preds_out.predictions
            probs       = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
            preds       = (probs >= 0.5).astype(int)

            f1  = f1_score(y_val, preds, pos_label=1, zero_division=0)
            auc = roc_auc_score(y_val, probs)
            ap  = average_precision_score(y_val, probs, pos_label=1)
            fold_scores.append({"f1": f1, "auc": auc, "ap": ap})

            print(f"Fold {fold+1} -> F1: {f1:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}")

            # Save fold model
            fold_best = f"{fold_dir}/best_model"
            trainer.save_model(fold_best)
            tokenizer.save_pretrained(fold_best)
            print(f"Fold {fold+1} model saved -> {fold_best}")

        # Print k-fold summary
        print(f"\n{'='*50}")
        print(f"K-Fold Summary ({args.kfold} folds)")
        print(f"{'='*50}")
        for metric in ["f1", "auc", "ap"]:
            vals = [s[metric] for s in fold_scores]
            print(f"  {metric.upper():3s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print(f"{'='*50}")
        print(f"\nBest fold by AP: fold {np.argmax([s['ap'] for s in fold_scores])+1}")
        print(f"Use that fold's checkpoint for submission, or retrain on full data.")

    # Standard single split mode
    else:
        print(f"\n{'='*50}")
        print(f"Standard Train/Val Split")
        print(f"{'='*50}")

        X_train, X_val, y_train, y_val = split_data(alltxts, alllabs)

        trainer = train_one(
            X_train, y_train, X_val, y_val,
            tokenizer, args.output_dir,
            args.epochs, args.batch_size, args.lr, args.augment, args.strategy
        )

        best_path = f"{args.output_dir}/best_model"
        trainer.save_model(best_path)
        tokenizer.save_pretrained(best_path)
        print(f"\nBest model saved -> {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CamemBERT speaker classifier")
    parser.add_argument("--fname",      type=str,   required=True,           help="Path to training corpus")
    parser.add_argument("--output_dir", type=str,   default="./checkpoints", help="Where to save checkpoints")
    parser.add_argument("--strategy",   type=str,   default="full",          help="Freeze strategy: full | head_only | top_layers")
    parser.add_argument("--epochs",     type=int,   default=3,               help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=16,              help="Per-device train batch size")
    parser.add_argument("--lr",         type=float, default=2e-5,            help="Learning rate")
    parser.add_argument("--kfold",      type=int,   default=1,               help="Number of k-fold splits (1 = no kfold)")
    parser.add_argument("--augment",    action="store_true",                 help="Augment Mitterrand training sentences")
    parser.add_argument("--context", action="store_true",                    help="Add neighboring sentence context around each sentence")
    args = parser.parse_args()

    train(args)
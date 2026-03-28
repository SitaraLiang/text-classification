from transformers import  AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import numpy as np
import evaluate
from sklearn.metrics import classification_report
import os
import pandas as pd
import csv
from nltk.corpus import stopwords
import string

def load_movies_1(path2data):
    alltxts = []
    labs = []
    cpt = 0
    stopwrds = set(stopwords.words('english'))
    for cl in sorted(os.listdir(path2data)):
        print("test :", cpt, cl)
        if cl == '.DS_Store':
            continue
        for f in os.listdir(path2data+cl):
            with open(path2data+cl+'/'+f, 'r', encoding='utf-8', errors='ignore') as file:
                txt = file.read().lower().split()
                txt = " ".join([word for word in txt if word not in stopwrds])
                alltxts.append(txt)
                labs.append(cpt)
        cpt += 1
    return alltxts, labs

#https://www.kaggle.com/code/nadzmiagthomas/distilbert-fine-tuning
#a bit outdated I think

#model_name = 'distilbert-base-uncased'
model_name = 'distilbert/distilroberta-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_movies(path2data): # 1 classe par r  pertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data): # parcours des fichiers d'un r  pertoire
        print("test :", cpt, cl)
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg r  pertoire = cht classe

    return alltxts,labs

path = "data/movies1000/"
#alltxts,alllabs = load_movies(path)
alltxts, alllabs = load_movies_1(path)

x_train, x_test, y_train, y_test = train_test_split(alltxts, alllabs, test_size=0.2, random_state=42)
print("taille x", len(x_train))
train_ds = Dataset.from_dict({"text": x_train, "label": y_train})
test_ds = Dataset.from_dict({"text": x_test, "label": y_test})
raw_datasets = DatasetDict({
    "train": train_ds,
    "test": test_ds
})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = raw_datasets.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"]
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc") # F1 and Accuracy
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="test-trainer",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    learning_rate=2e-5,
    save_strategy="epoch",
    lr_scheduler_type="linear",
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)

trainer.train()

with open("data/test/testSentiment.txt", 'r', encoding='utf-8') as file:
    #lines = file.readlines()
    lines = [line.strip() for line in file if line.strip()]

trainer.evaluate()
#results = trainer.predict(tokenized_datasets["test"])
x = [tokenizer(line, truncation=True, max_length=512) for line in lines]
test_dataset = Dataset.from_list(x)
results = trainer.predict(test_dataset)
y_pred = np.argmax(results.predictions, axis=-1)
print(y_pred, y_pred.shape)
print(tokenized_datasets["test"].shape)

#y_true = results.label_ids

#report = classification_report(y_true, y_pred, digits=3, output_dict=True)
#df = pd.DataFrame(report).transpose()
#df.to_csv("out.csv")
preds = ["P" if p == 1 else "N" for p in y_pred]
print(preds)
with open("out_pred.csv", 'w', newline='', encoding='utf-8') as destination:
    writer = csv.writer(destination)
    for row in preds:
        writer.writerow(row)

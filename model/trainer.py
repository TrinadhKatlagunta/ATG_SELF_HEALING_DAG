# model/trainer.py

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np
import evaluate

# Config
MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "./model/model_files/"
NUM_LABELS = 6  # Emotion dataset has 6 labels

def load_data():
    dataset = load_dataset("emotion")
    return dataset

def preprocess_data(dataset, tokenizer):
    def preprocess(example):
        return tokenizer(example["text"], truncation=True)
    
    encoded = dataset.map(preprocess, batched=True)
    return encoded

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train():
    print("[INFO] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    dataset = load_data()
    encoded = preprocess_data(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=1,
        do_train=True,
        do_eval=True,
        logging_steps=100,
        # Removed `load_best_model_at_end` to avoid strategy mismatch error
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model to:", SAVE_PATH)
    trainer.save_model(SAVE_PATH)

if __name__ == "__main__":
    train()
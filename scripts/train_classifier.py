"""
Fine-Tune XLM-RoBERTa for Document Classification.

Usage:
    python scripts/train_classifier.py

Reads:  data/processed/cls_train.csv, cls_val.csv
Writes: models/classifier/ (best model by validation f1_macro)

Classifies into 8 categories:
    invoice, receipt, id_card, legal_document,
    medical_report, bank_statement, govt_form, letter
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CATEGORIES = [
    "invoice", "receipt", "id_card", "legal_document",
    "medical_report", "bank_statement", "govt_form", "letter",
]
LABEL2ID = {cat: i for i, cat in enumerate(CATEGORIES)}
ID2LABEL = {i: cat for i, cat in enumerate(CATEGORIES)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics():
    from sklearn.metrics import f1_score, classification_report

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)

        # Log per-category report
        report = classification_report(
            labels, predictions,
            target_names=CATEGORIES,
            zero_division=0,
        )
        logger.info("\n%s", report)

        return {
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from src.utils.config import get_config

    cfg = get_config()["training"]["classifier"]
    base_model = cfg["base_model"]
    output_dir = cfg["output_dir"]
    max_length = cfg["max_length"]
    has_gpu = torch.cuda.is_available()

    data_dir = Path("data/processed")
    train_path = data_dir / "cls_train.csv"
    val_path = data_dir / "cls_val.csv"

    if not train_path.exists():
        logger.error("Training data not found: %s", train_path)
        logger.error("Run: python scripts/generate_synthetic_data.py first")
        sys.exit(1)

    logger.info("Loading training data...")
    train_df = pd.read_csv(train_path).dropna(subset=["text", "label"])
    val_df = pd.read_csv(val_path).dropna(subset=["text", "label"])

    # Filter to known categories
    train_df = train_df[train_df["label"].isin(CATEGORIES)].reset_index(drop=True)
    val_df = val_df[val_df["label"].isin(CATEGORIES)].reset_index(drop=True)

    logger.info("Train: %d samples | Val: %d samples", len(train_df), len(val_df))

    # Convert labels to IDs
    train_labels = [LABEL2ID[l] for l in train_df["label"]]
    val_labels = [LABEL2ID[l] for l in val_df["label"]]

    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("Building datasets...")
    train_dataset = ClassificationDataset(
        list(train_df["text"]), train_labels, tokenizer, max_length
    )
    val_dataset = ClassificationDataset(
        list(val_df["text"]), val_labels, tokenizer, max_length
    )

    logger.info("Loading model: %s (num_labels=%d)", base_model, len(CATEGORIES))
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        fp16=cfg["fp16"] and has_gpu,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        save_total_limit=cfg["save_total_limit"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )

    logger.info("Starting Classifier fine-tuning on %s...", "GPU" if has_gpu else "CPU")
    trainer.train()

    logger.info("Saving best model to %s/", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Evaluating on validation set...")
    metrics = trainer.evaluate()
    logger.info("Final metrics: %s", metrics)
    logger.info("Classifier training complete! Set classifier.use_finetuned: true in pipeline_config.yaml")


if __name__ == "__main__":
    main()

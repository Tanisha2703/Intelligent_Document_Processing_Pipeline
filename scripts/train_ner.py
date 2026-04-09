"""
Fine-Tune XLM-RoBERTa for Named Entity Recognition.

Usage:
    python scripts/train_ner.py

Reads:  data/processed/ner_train.conll, ner_val.conll
Writes: models/ner/ (best model by validation overall_f1)

Key design decisions:
  - Subword label alignment: only the FIRST subword of each word gets the
    actual NER label; all subsequent subwords get -100 (ignored in loss).
    This is CRITICAL for correct NER training with subword tokenizers.
  - fp16=True requires a CUDA GPU; set to False for CPU training.
  - early_stopping_patience=3 prevents overfitting on synthetic data.
"""
import logging
import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoNLL reader
# ---------------------------------------------------------------------------

def read_conll(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    """Read CoNLL-format file → (list_of_token_lists, list_of_label_lists)."""
    all_tokens, all_labels = [], []
    cur_tokens, cur_labels = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if cur_tokens:
                    all_tokens.append(cur_tokens)
                    all_labels.append(cur_labels)
                    cur_tokens, cur_labels = [], []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    cur_tokens.append(parts[0])
                    cur_labels.append(parts[-1])

    if cur_tokens:
        all_tokens.append(cur_tokens)
        all_labels.append(cur_labels)

    return all_tokens, all_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NERDataset(torch.utils.data.Dataset):
    """Token classification dataset with subword alignment."""

    def __init__(self, tokens_list, labels_list, tokenizer, label2id, max_length=256):
        self.encodings = []
        skipped = 0
        for tokens, labels in zip(tokens_list, labels_list):
            enc = self._encode(tokens, labels, tokenizer, label2id, max_length)
            if enc:
                self.encodings.append(enc)
            else:
                skipped += 1
        if skipped:
            logger.debug("Skipped %d empty sentences", skipped)

    @staticmethod
    def _encode(tokens, labels, tokenizer, label2id, max_length):
        """Tokenize words and align labels to subword tokens."""
        if not tokens:
            return None

        # tokenize_is_split_into_words handles word-level input
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # Special token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                # First subword of a new word → assign real label
                label_str = labels[word_id] if word_id < len(labels) else "O"
                aligned_labels.append(label2id.get(label_str, 0))
            else:
                # Continuation subword → ignore in loss
                aligned_labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(id2label):
    """Return a compute_metrics function for seqeval entity-level F1."""
    import evaluate
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_row, label_row in zip(predictions, labels):
            sent_labels, sent_preds = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                sent_labels.append(id2label[l])
                sent_preds.append(id2label[p])
            true_labels.append(sent_labels)
            true_preds.append(sent_preds)

        result = seqeval.compute(
            predictions=true_preds, references=true_labels, zero_division=0
        )

        # Log per-entity F1
        for key, val in result.items():
            if isinstance(val, dict) and "f1" in val:
                logger.info("  %-20s F1=%.4f P=%.4f R=%.4f",
                            key, val["f1"], val["precision"], val["recall"])

        return {
            "overall_precision": result["overall_precision"],
            "overall_recall": result["overall_recall"],
            "overall_f1": result["overall_f1"],
            "overall_accuracy": result["overall_accuracy"],
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from src.utils.config import get_config
    from src.ner.entities import LABEL2ID, ID2LABEL, NUM_LABELS

    cfg = get_config()["training"]["ner"]
    base_model = cfg["base_model"]
    output_dir = cfg["output_dir"]
    max_length = cfg["max_length"]
    has_gpu = torch.cuda.is_available()

    data_dir = Path("data/processed")
    train_path = data_dir / "ner_train.conll"
    val_path = data_dir / "ner_val.conll"

    if not train_path.exists():
        logger.error("Training data not found: %s", train_path)
        logger.error("Run: python scripts/generate_synthetic_data.py first")
        sys.exit(1)

    logger.info("Loading training data from %s", train_path)
    train_tokens, train_labels = read_conll(train_path)
    val_tokens, val_labels = read_conll(val_path)
    logger.info("Train: %d sentences | Val: %d sentences", len(train_tokens), len(val_tokens))

    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("Building datasets...")
    train_dataset = NERDataset(train_tokens, train_labels, tokenizer, LABEL2ID, max_length)
    val_dataset = NERDataset(val_tokens, val_labels, tokenizer, LABEL2ID, max_length)
    logger.info("Train: %d samples | Val: %d samples", len(train_dataset), len(val_dataset))

    logger.info("Loading model: %s (num_labels=%d)", base_model, NUM_LABELS)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
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
        compute_metrics=build_compute_metrics(ID2LABEL),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )

    logger.info("Starting NER fine-tuning on %s...", "GPU" if has_gpu else "CPU")
    trainer.train()

    logger.info("Saving best model to %s/", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Evaluating on validation set...")
    metrics = trainer.evaluate()
    logger.info("Final metrics: %s", metrics)
    logger.info("NER training complete! Set ner.use_finetuned: true in pipeline_config.yaml")


if __name__ == "__main__":
    main()

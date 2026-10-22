"""
Assignment 5: Binary Text Classification Using Pretrained BERT
Based on NLP_ex_5_175.pdf (UCS3564 - NLP Lab).

SOURCE VS RECONSTRUCTION
------------------------
The PDF's source-code listing is split across pages and contains an omitted
benchmark-inference loop. This file reconstructs those pieces into one runnable
program. The dataset, 1,200-review balanced sample, seed 42, 70/15/15 split,
max length 128, batch size 16, TF-IDF + Logistic Regression baseline, BERT
evaluation metrics, confusion-matrix image, and linguistic benchmark themes
follow the PDF.

Implementation detail: the PDF's printed code creates BertForSequenceClassification
from a BertConfig, which initializes a model architecture without loading the
pretrained weights. This reconstruction uses
BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
to match the assignment's stated requirement of fine-tuning pretrained BERT.
The PDF's 4e-4 learning rate is retained as a configurable constant below, but
it is unusually high for BERT fine-tuning; if training is unstable, try 2e-5.
"""

import os
import re
import random
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Configuration (PDF values unless noted)
# ---------------------------------------------------------------------------
DATASET_PATH = "IMDB_Dataset.csv"
MODEL_NAME = "bert-base-uncased"
SAMPLE_SIZE = 1200
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 4e-4  # PDF value; try 2e-5 if training is unstable.
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
RANDOM_SEED = 42
CONFUSION_MATRIX_FILE = "confusion_matrix.png"


def set_seed(seed=42):
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Text cleaning (PDF Phase 2)
# ---------------------------------------------------------------------------
def clean_text(text):
    """Remove HTML tags and normalize whitespace while preserving punctuation."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 2. Dataset wrapper and WordPiece tokenization (PDF Phase 3)
# ---------------------------------------------------------------------------
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3. Evaluation helper (PDF Phase 7)
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return {
        "loss": total_loss / max(1, len(dataloader)),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
        "confusion_matrix": confusion_matrix(
            all_labels, all_preds, labels=[0, 1]
        ),
        "predictions": np.asarray(all_preds),
        "ground_truth": np.asarray(all_labels),
    }


def plot_and_save_confusion_matrix(cm, filename=CONFUSION_MATRIX_FILE):
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar=True,
    )
    plt.title("BERT Confusion Matrix - IMDB Test Set", fontsize=13, pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Ground Truth Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def print_metrics(title, metrics):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)
    print(f"Accuracy  : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1']:.4f}")
    print(f"Macro F1  : {metrics['f1_macro']:.4f}")
    print("Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(metrics["confusion_matrix"])
    print("\nDetailed Classification Report:")
    print(
        classification_report(
            metrics["ground_truth"],
            metrics["predictions"],
            labels=[0, 1],
            target_names=["Negative", "Positive"],
            zero_division=0,
        )
    )


# ---------------------------------------------------------------------------
# 4. Inference helpers and linguistic benchmark (PDF Phase 8)
# ---------------------------------------------------------------------------
ASSIGNMENT_REVIEWS = [
    (
        "The movie was an absolute masterpiece with brilliant performances "
        "and a gripping story.",
        "Positive",
    ),
    (
        "I found the plot boring and the acting was extremely weak throughout.",
        "Negative",
    ),
    (
        "A delightful film that kept me engaged from start to finish.",
        "Positive",
    ),
    (
        "This was one of the worst movies I have ever sat through.",
        "Negative",
    ),
]

# Ten custom linguistic stress tests adapted from the PDF's page 11 examples.
STRESS_TESTS = [
    ("The movie was not bad at all, in fact it turned out to be surprisingly enjoyable.", "Positive", "Negation"),
    ("This was definitely not the cinematic masterpiece that enthusiastic critics claimed.", "Negative", "Negation"),
    ("I did not hate the direction, but it certainly was not inspiring either.", "Negative", "Mixed sentiment"),
    ("Oh wonderful, another two hours of my precious life wasted on cinematic genius.", "Negative", "Sarcasm"),
    ("Truly remarkable: a two-hour film where absolutely nothing interesting occurs.", "Negative", "Sarcasm"),
    ("Although the visual effects were breathtaking, the script was atrocious.", "Negative", "Concession"),
    ("Despite the slow start and weak dialogue, the lead actor completely saved the film.", "Positive", "Concession"),
    ("I expected a disaster, but the director pulled off a complete triumph.", "Positive", "Contrast"),
    ("I would rather watch paint dry than sit through this torturous disaster again.", "Negative", "Idiom"),
    ("Not once did my attention waver; every sequence was executed with precision.", "Positive", "Inverted syntax"),
]


def predict_bert(texts, model, tokenizer, device, max_length=128, batch_size=16):
    model.eval()
    predictions = []
    confidences = []

    for start in range(0, len(texts), batch_size):
        batch_texts = [clean_text(t) for t in texts[start:start + batch_size]]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {key: value.to(device) for key, value in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
        predictions.extend(pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())

    return predictions, confidences


def run_benchmarks(model, tokenizer, bow_model, tfidf, device):
    cases = [(text, truth, "Assignment sample") for text, truth in ASSIGNMENT_REVIEWS]
    cases.extend(STRESS_TESTS)

    texts = [item[0] for item in cases]
    bert_preds, bert_confs = predict_bert(texts, model, tokenizer, device)
    bow_preds = bow_model.predict(tfidf.transform(texts))

    rows = []
    for i, ((text, truth, category), bert_pred, conf, bow_pred) in enumerate(
        zip(cases, bert_preds, bert_confs, bow_preds), start=1
    ):
        rows.append({
            "ID": i,
            "Category": category,
            "Review": text,
            "Ground Truth": truth,
            "BERT Prediction": "Positive" if bert_pred == 1 else "Negative",
            "BERT Confidence (%)": round(conf * 100, 2),
            "BoW Prediction": "Positive" if int(bow_pred) == 1 else "Negative",
        })

    print("\n" + "=" * 62)
    print("INFERENCE & COMPARATIVE LINGUISTIC BENCHMARK")
    print("=" * 62)
    for row in rows:
        print(f"\n[Case {row['ID']}] {row['Category']}")
        print("Review:", row["Review"])
        print("Ground truth:", row["Ground Truth"])
        print(
            f"BERT: {row['BERT Prediction']} "
            f"(confidence: {row['BERT Confidence (%)']:.2f}%)"
        )
        print("BoW :", row["BoW Prediction"])

    pd.DataFrame(rows).to_csv("benchmark_predictions.csv", index=False)
    print("\nSaved benchmark results to benchmark_predictions.csv")


# ---------------------------------------------------------------------------
# 5. Main execution
# ---------------------------------------------------------------------------
def main():
    print("UCS3564 - NLP Assignment 5: BERT Binary Text Classification")
    print("=" * 62)
    print(f"Execution device: {DEVICE}")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Could not find '{DATASET_PATH}'. Put IMDB_Dataset.csv in the "
            "same folder as this script. Expected columns: 'review' and "
            "'sentiment'. Download the IMDB 50K dataset specified in the PDF."
        )

    # 5.1 Dataset ingestion and labels (PDF)
    df_raw = pd.read_csv(DATASET_PATH)
    required = {"review", "sentiment"}
    if not required.issubset(df_raw.columns):
        raise ValueError(
            f"CSV must contain columns {sorted(required)}; found {list(df_raw.columns)}"
        )

    df_raw = df_raw.dropna(subset=["review", "sentiment"]).copy()
    df_raw["sentiment"] = df_raw["sentiment"].astype(str).str.lower().str.strip()
    df_raw["label"] = df_raw["sentiment"].map({"negative": 0, "positive": 1})
    df_raw = df_raw.dropna(subset=["label"]).copy()
    df_raw["label"] = df_raw["label"].astype(int)

    print(f"\nRaw dataset rows: {len(df_raw)}")

    # PDF samples 600 positive + 600 negative reviews.
    per_class = SAMPLE_SIZE // 2
    available = df_raw["label"].value_counts()
    if len(available) < 2 or min(available.get(0, 0), available.get(1, 0)) < per_class:
        raise ValueError(
            f"Need at least {per_class} reviews from each class for SAMPLE_SIZE={SAMPLE_SIZE}."
        )

    df_pos = df_raw[df_raw["label"] == 1].sample(n=per_class, random_state=RANDOM_SEED)
    df_neg = df_raw[df_raw["label"] == 0].sample(n=per_class, random_state=RANDOM_SEED)
    df_sampled = (
        pd.concat([df_pos, df_neg])
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    df_sampled["cleaned_review"] = df_sampled["review"].apply(clean_text)

    print(f"Balanced sample rows: {len(df_sampled)}")
    print(df_sampled["sentiment"].value_counts().to_string())

    # 5.2 Stratified 70/15/15 split (PDF)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        df_sampled["cleaned_review"].tolist(),
        df_sampled["label"].tolist(),
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df_sampled["label"].tolist(),
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )
    print("\nStratified dataset split:")
    print(f"Training samples  (70%): {len(X_train_raw)}")
    print(f"Validation samples (15%): {len(X_val_raw)}")
    print(f"Testing samples   (15%): {len(X_test_raw)}")

    # 5.3 TF-IDF + Logistic Regression baseline (PDF Phase 5)
    print("\nTraining baseline TF-IDF + Logistic Regression...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 1),
    )
    bow_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=200)
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    bow_model.fit(X_train_tfidf, y_train)
    bow_test_preds = bow_model.predict(tfidf.transform(X_test_raw))
    bow_precision, bow_recall, bow_f1, _ = precision_recall_fscore_support(
        y_test, bow_test_preds, average="binary", zero_division=0
    )
    bow_metrics = {
        "accuracy": accuracy_score(y_test, bow_test_preds),
        "precision": bow_precision,
        "recall": bow_recall,
        "f1": bow_f1,
        "f1_macro": precision_recall_fscore_support(
            y_test, bow_test_preds, average="macro", zero_division=0
        )[2],
        "confusion_matrix": confusion_matrix(y_test, bow_test_preds, labels=[0, 1]),
        "predictions": np.asarray(bow_test_preds),
        "ground_truth": np.asarray(y_test),
    }
    print_metrics("BASELINE TF-IDF + LOGISTIC REGRESSION TEST RESULTS", bow_metrics)

    # 5.4 Tokenizer and DataLoaders
    print("\nLoading BERT WordPiece tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        IMDBDataset(X_train_raw, y_train, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        IMDBDataset(X_val_raw, y_val, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        IMDBDataset(X_test_raw, y_test, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # 5.5 Pretrained BERT with binary classification head
    print("\nLoading pretrained BERT model with binary classification head...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"BERT model parameters: {total_params:,}")

    # 5.6 Training loop with AdamW, warmup scheduler, gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    print(f"\nCommencing BERT training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += outputs.loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

    # 5.7 Final held-out test evaluation
    print("\nEvaluating trained BERT on held-out test set...")
    bert_metrics = evaluate_model(model, test_loader, DEVICE)
    print_metrics("BERT MODEL TEST PERFORMANCE SUMMARY", bert_metrics)
    plot_and_save_confusion_matrix(
        bert_metrics["confusion_matrix"], CONFUSION_MATRIX_FILE
    )
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")

    print("\n" + "=" * 62)
    print("BASELINE VS BERT")
    print("=" * 62)
    print(f"{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for name, metrics in [
        ("TF-IDF + Logistic Regression", bow_metrics),
        ("Fine-tuned BERT", bert_metrics),
    ]:
        print(
            f"{name:<35} {metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )

    # 5.8 Mandatory sample reviews + 10 linguistic stress tests
    run_benchmarks(model, tokenizer, bow_model, tfidf, DEVICE)

    # Save the fine-tuned model/tokenizer for reuse.
    output_dir = "bert_imdb_finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved fine-tuned model and tokenizer to: {output_dir}")
    print("Finished Assignment 5.")


if __name__ == "__main__":
    main()

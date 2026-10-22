"""
UCS3564 NLP Lab — Assignment 4
Neural Machine Translation using Seq2Seq and Attention Models

RECONSTRUCTION NOTE
-------------------
The PDF describes the complete pipeline, but it does not provide a complete
source-code listing. This runnable script reconstructs the missing code from
the report's described workflow and console output. It includes:
  1) Unicode NFKC normalization and punctuation-aware cleaning
  2) A word-level vocabulary and custom PyTorch BiGRU Seq2Seq with Bahdanau attention
  3) Training on a configurable subset of the English–French CSV
  4) Pretrained MarianMT English→French and English→German translation
  5) BLEU-1, BLEU-4, and SacreBLEU evaluation
  6) Attention heatmap visualizations (MarianMT decoder cross-attention)
  7) Interactive translation CLI and qualitative error analysis

The custom Seq2Seq is a compact demonstration model; MarianMT is the primary
pretrained translation model. Numeric scores can differ from those printed in
the PDF because model weights, random sampling, package versions, and training
runs differ. Attention tensor formats can also vary across Transformers versions.
"""

import os
import re
import html
import random
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer

try:
    import sacrebleu
except ImportError:
    sacrebleu = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("archive", "eng_-french.csv")
MODEL_FR_NAME = "Helsinki-NLP/opus-mt-en-fr"
MODEL_DE_NAME = "Helsinki-NLP/opus-mt-en-de"

SEED = 42
CUSTOM_SAMPLE_SIZE = 5000
CUSTOM_EPOCHS = 3
CUSTOM_BATCH_SIZE = 64
CUSTOM_MAX_VOCAB = 12000
CUSTOM_MAX_LEN = 30
CUSTOM_EMBED_DIM = 128
CUSTOM_HIDDEN_DIM = 128
CUSTOM_LEARNING_RATE = 0.001
MAX_MARIAN_INPUT_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text):
    """Unicode NFKC normalization; preserve French accents and normalize spaces."""
    text = unicodedata.normalize("NFKC", str(text))
    text = html.unescape(text)
    text = text.replace("\u202f", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_word_model(text):
    """Normalize punctuation spacing for the custom word-level model."""
    text = normalize_text(text).lower()
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# BLEU metrics
# ---------------------------------------------------------------------------
def bleu_scores(reference, hypothesis):
    """Return BLEU-1, BLEU-4, and SacreBLEU scores on a 0–100 scale."""
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    # NLTK BLEU is used as a fallback; SacreBLEU is preferred when installed.
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        smooth = SmoothingFunction().method1
        bleu1 = 100.0 * sentence_bleu(
            [ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth
        )
        bleu4 = 100.0 * sentence_bleu(
            [ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth
        )
    except Exception:
        bleu1, bleu4 = 0.0, 0.0

    if sacrebleu is not None:
        sacre = sacrebleu.sentence_bleu(hypothesis, [reference]).score
    else:
        sacre = bleu4
    return bleu1, bleu4, sacre


# ---------------------------------------------------------------------------
# Custom word-level vocabulary and dataset
# ---------------------------------------------------------------------------
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


class Vocabulary:
    def __init__(self, max_size=CUSTOM_MAX_VOCAB):
        self.max_size = max_size
        self.itos = [PAD, SOS, EOS, UNK]
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def build(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(clean_for_word_model(text).split())
        for token, _ in counts.most_common(self.max_size - len(self.itos)):
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, text, max_len=CUSTOM_MAX_LEN):
        tokens = clean_for_word_model(text).split()[: max_len - 2]
        ids = [self.stoi[SOS]]
        ids.extend(self.stoi.get(token, self.stoi[UNK]) for token in tokens)
        ids.append(self.stoi[EOS])
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        words = []
        for idx in ids:
            token = self.itos[int(idx)]
            if token == EOS:
                break
            if token not in (PAD, SOS):
                words.append(token)
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?;:)])", r"\1", text)
        text = re.sub(r"([(])\s+", r"\1", text)
        text = text.replace(" ' ", "'")
        return text


class ParallelDataset(Dataset):
    def __init__(self, source_texts, target_texts, src_vocab, tgt_vocab):
        self.source_texts = list(source_texts)
        self.target_texts = list(target_texts)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        return (
            self.src_vocab.encode(self.source_texts[idx]),
            self.tgt_vocab.encode(self.target_texts[idx]),
        )


def collate_parallel(batch):
    sources, targets = zip(*batch)
    return (
        pad_sequence(sources, batch_first=True, padding_value=0),
        pad_sequence(targets, batch_first=True, padding_value=0),
    )


# ---------------------------------------------------------------------------
# Custom Seq2Seq: Bidirectional GRU encoder + Bahdanau additive attention
# ---------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, attn_dim, bias=False)
        self.dec_proj = nn.Linear(dec_dim, attn_dim, bias=False)
        self.energy = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        # decoder_hidden: [batch, dec_dim]
        # encoder_outputs: [batch, source_len, enc_dim]
        scores = self.energy(
            torch.tanh(
                self.enc_proj(encoder_outputs)
                + self.dec_proj(decoder_hidden).unsqueeze(1)
            )
        ).squeeze(-1)
        scores = scores.masked_fill(~src_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.enc_dim = hidden_dim * 2
        self.bridge = nn.Linear(self.enc_dim, hidden_dim)
        self.attention = BahdanauAttention(self.enc_dim, hidden_dim, hidden_dim)
        self.decoder = nn.GRU(embed_dim + self.enc_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(embed_dim + self.enc_dim + hidden_dim, tgt_vocab_size)

    def encode(self, src):
        src_mask = src.ne(0)
        embedded = self.src_embedding(src)
        enc_outputs, hidden = self.encoder(embedded)
        # Combine final forward and backward states into one decoder initial state.
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        dec_hidden = torch.tanh(self.bridge(final)).unsqueeze(0)
        return enc_outputs, dec_hidden, src_mask

    def forward(self, src, tgt):
        enc_outputs, hidden, src_mask = self.encode(src)
        logits_steps = []
        # Teacher forcing: predict target token t from target token t-1.
        for t in range(tgt.size(1) - 1):
            prev_token = tgt[:, t]
            emb = self.tgt_embedding(prev_token)
            context, _ = self.attention(hidden[-1], enc_outputs, src_mask)
            dec_input = torch.cat((emb, context), dim=1).unsqueeze(1)
            dec_out, hidden = self.decoder(dec_input, hidden)
            combined = torch.cat((dec_out.squeeze(1), context, emb), dim=1)
            logits_steps.append(self.output(combined))
        return torch.stack(logits_steps, dim=1) if logits_steps else None

    @torch.no_grad()
    def translate(self, source_text, src_vocab, tgt_vocab, max_len=CUSTOM_MAX_LEN):
        self.eval()
        src = src_vocab.encode(source_text).unsqueeze(0).to(next(self.parameters()).device)
        enc_outputs, hidden, src_mask = self.encode(src)
        current = torch.tensor([[tgt_vocab.stoi[SOS]]], device=src.device)
        output_ids = []
        for _ in range(max_len):
            emb = self.tgt_embedding(current[:, -1])
            context, _ = self.attention(hidden[-1], enc_outputs, src_mask)
            dec_input = torch.cat((emb, context), dim=1).unsqueeze(1)
            dec_out, hidden = self.decoder(dec_input, hidden)
            logits = self.output(torch.cat((dec_out.squeeze(1), context, emb), dim=1))
            next_id = int(logits.argmax(dim=1).item())
            if next_id == tgt_vocab.stoi[EOS]:
                break
            output_ids.append(next_id)
            current = torch.cat(
                (current, torch.tensor([[next_id]], device=src.device)), dim=1
            )
        return tgt_vocab.decode(output_ids)


def load_parallel_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the Kaggle English–French translation CSV and place it at "
            f"'{path}'. The CSV must contain English and French columns."
        )

    df = pd.read_csv(path)
    # Handle common column naming conventions.
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    source_col = next(
        (lower_to_original[k] for k in ("english", "en", "source", "english sentence")
         if k in lower_to_original),
        None,
    )
    target_col = next(
        (lower_to_original[k] for k in ("french", "fr", "target", "french sentence")
         if k in lower_to_original),
        None,
    )
    if source_col is None or target_col is None:
        if len(df.columns) >= 2:
            source_col, target_col = df.columns[:2]
            print(
                f"[WARN] Could not identify language columns by name; using "
                f"'{source_col}' → '{target_col}'. Please verify the CSV column order."
            )
        else:
            raise ValueError("CSV needs at least two columns: English and French.")

    df = df[[source_col, target_col]].dropna()
    df.columns = ["english", "french"]
    df["english"] = df["english"].map(normalize_text)
    df["french"] = df["french"].map(normalize_text)
    df = df[(df["english"].str.len() > 0) & (df["french"].str.len() > 0)]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def train_custom_seq2seq(df):
    print("\n" + "=" * 66)
    print("BUILDING CUSTOM SEQ2SEQ WITH BAHDANAU ATTENTION (PYTORCH)")
    print("=" * 66)

    sample_n = min(CUSTOM_SAMPLE_SIZE, len(df))
    sampled = df.sample(n=sample_n, random_state=SEED).reset_index(drop=True)
    src_texts = sampled["english"].tolist()
    tgt_texts = sampled["french"].tolist()

    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build(src_texts)
    tgt_vocab.build(tgt_texts)

    print(f"[INFO] Selected subset: {sample_n} sentence pairs")
    print(f"[INFO] Vocabulary Sizes - English: {len(src_vocab.itos)}, French: {len(tgt_vocab.itos)}")

    dataset = ParallelDataset(src_texts, tgt_texts, src_vocab, tgt_vocab)
    loader = DataLoader(
        dataset, batch_size=CUSTOM_BATCH_SIZE, shuffle=True,
        collate_fn=collate_parallel,
    )

    model = Seq2SeqAttention(
        len(src_vocab.itos), len(tgt_vocab.itos),
        embed_dim=CUSTOM_EMBED_DIM, hidden_dim=CUSTOM_HIDDEN_DIM,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CUSTOM_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"[INFO] Training custom Bahdanau Seq2Seq model ({CUSTOM_EPOCHS} epochs)...")
    for epoch in range(1, CUSTOM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, tgt)
            if logits is None:
                continue
            gold = tgt[:, 1:1 + logits.size(1)]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch}/{CUSTOM_EPOCHS}] Average Loss: {total_loss / max(1, len(loader)):.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "src_vocab": src_vocab.itos,
            "tgt_vocab": tgt_vocab.itos,
        },
        "custom_seq2seq_attention.pt",
    )
    print("[INFO] Saved custom model to custom_seq2seq_attention.pt")
    return model, src_vocab, tgt_vocab


# ---------------------------------------------------------------------------
# Pretrained MarianMT translation and attention visualization
# ---------------------------------------------------------------------------
def load_marian(model_name):
    print(f"[INFO] Loading pretrained model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def marian_translate(texts, tokenizer, model, max_length=MAX_MARIAN_INPUT_LEN):
    enc = tokenizer(
        [normalize_text(t) for t in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {key: value.to(DEVICE) for key, value in enc.items()}
    generated = model.generate(**enc, max_length=max_length)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


@torch.no_grad()
def save_attention_heatmap(text, tokenizer, model, filename, title):
    """Save averaged last-layer cross-attention from a MarianMT forward pass."""
    enc = tokenizer(
        normalize_text(text), return_tensors="pt", truncation=True,
        max_length=MAX_MARIAN_INPUT_LEN,
    )
    enc = {key: value.to(DEVICE) for key, value in enc.items()}
    generated = model.generate(**enc, max_length=MAX_MARIAN_INPUT_LEN)
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=generated)
    outputs = model(
        **enc,
        decoder_input_ids=decoder_input_ids,
        output_attentions=True,
        return_dict=True,
    )

    cross = outputs.cross_attentions
    if not cross:
        print(f"[WARN] Cross-attention unavailable; skipped {filename}")
        return

    # Last decoder layer: [batch, heads, target_len, source_len]
    attn = cross[-1][0].mean(dim=0).detach().cpu().numpy()
    source_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    target_tokens = tokenizer.convert_ids_to_tokens(generated[0])
    attn = attn[:len(target_tokens), :len(source_tokens)]

    plt.figure(figsize=(max(9, len(source_tokens) * 0.45),
                        max(5, len(target_tokens) * 0.35)), dpi=180)
    sns.heatmap(
        attn, cmap="Blues", xticklabels=source_tokens,
        yticklabels=target_tokens, cbar=True,
    )
    plt.xlabel("Source English Tokens")
    plt.ylabel("Generated French Tokens")
    plt.title(title)
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Attention heatmap saved: {filename}")


# ---------------------------------------------------------------------------
# Required benchmark sentences and custom linguistic stress tests (PDF)
# ---------------------------------------------------------------------------
BENCHMARKS = [
    {
        "english": "The weather in Paris is usually cold during January, but it is warm in July.",
        "reference": "Le temps à Paris est généralement froid en janvier, mais il fait chaud en juillet.",
    },
    {
        "english": "She likes to read books, watch movies, and travel to new places.",
        "reference": "Elle aime lire des livres, regarder des films et voyager dans de nouveaux endroits.",
    },
    {
        "english": "The new employee works hard every day and completes his tasks on time.",
        "reference": "Le nouvel employé travaille dur chaque jour et termine ses tâches à temps.",
    },
]

CUSTOM_SENTENCES = [
    ("Although artificial intelligence has advanced rapidly in recent years, many challenges remain.", "Long sentence"),
    ("The professor reminded all students that the final project submission deadline cannot be extended.", "Long sentence"),
    ("Every morning before heading to the laboratory, the researchers carefully review their experimental results.", "Long sentence"),
    ("The patient was diagnosed with severe cardiomyopathy caused by chronic mitochondrial dysfunction.", "Medicine"),
    ("Quantum entanglement enables superdense coding and secure cryptographic key distribution.", "Quantum computing"),
    ("Deforestation severely accelerates biodiversity loss and disrupts regional precipitation patterns.", "Ecology"),
    ("It is raining cats and dogs outside, so remember to take your umbrella.", "Idiom"),
    ("Actions speak louder than words when building trust with customers.", "Idiom"),
    ("I am glad that you were able to visit us despite the terrible storm.", "Subjunctive"),
    ("Could you please tell me where the nearest train station is located?", "Polite question"),
    ("The ancient manuscript was discovered by archaeologists buried beneath centuries of sediment.", "Passive voice"),
    ("If you prepare thoroughly and practice every day, you will pass the examination easily.", "Conditional"),
]


def evaluate_sentence_set(items, tokenizer_fr, model_fr, label):
    sources = [item[0] if isinstance(item, tuple) else item["english"] for item in items]
    refs = [item[1] if isinstance(item, tuple) else item["reference"] for item in items]
    hypotheses = marian_translate(sources, tokenizer_fr, model_fr)
    rows = []
    print("\n" + "=" * 66)
    print(label)
    print("=" * 66)
    for idx, (src, ref, hyp) in enumerate(zip(sources, refs, hypotheses), start=1):
        b1, b4, sacre = bleu_scores(ref, hyp)
        category = items[idx - 1][1] if isinstance(items[idx - 1], tuple) else "Mandatory benchmark"
        rows.append({
            "ID": idx,
            "Category": category,
            "English": src,
            "Reference French": ref,
            "Prediction": hyp,
            "BLEU-1": round(b1, 2),
            "BLEU-4": round(b4, 2),
            "SacreBLEU": round(sacre, 2),
        })
        print(f"\n[{idx}] {category}")
        print("EN :", src)
        print("REF:", ref)
        print("HYP:", hyp)
        print(f"BLEU-1: {b1:.2f} | BLEU-4: {b4:.2f} | SacreBLEU: {sacre:.2f}")
    return rows


def run_german_comparison(tokenizer_de, model_de):
    print("\n" + "=" * 66)
    print("BONUS TASK (a): ENGLISH → GERMAN COMPARISON")
    print("=" * 66)
    sources = [x["english"] for x in BENCHMARKS]
    translations = marian_translate(sources, tokenizer_de, model_de)
    for i, (src, de) in enumerate(zip(sources, translations), start=1):
        print(f"\nSample {i}")
        print("EN:", src)
        print("DE:", de)


def error_analysis(tokenizer_fr, model_fr):
    print("\n" + "=" * 66)
    print("BONUS TASK (d): QUALITATIVE TRANSLATION ERROR ANALYSIS")
    print("=" * 66)
    text = "It is raining cats and dogs outside, so remember to take your umbrella."
    hypothesis = marian_translate([text], tokenizer_fr, model_fr)[0]
    print("Error/challenge sentence:", text)
    print("Model output:", hypothesis)
    print("Expected idiomatic French: Il pleut des cordes dehors, alors n'oublie pas de prendre ton parapluie.")
    print(
        "Analysis: Compare whether the translation preserves the idiomatic meaning "
        "or translates the words literally. This is a qualitative diagnosis; the "
        "script does not assume the model will make the exact error shown in the PDF."
    )


def interactive_cli(tokenizer_fr, model_fr, tokenizer_de=None, model_de=None):
    print("\n" + "=" * 66)
    print("BONUS TASK (c): INTERACTIVE TRANSLATION CLI")
    print("Enter an English sentence. Type 'quit' to exit.")
    print("=" * 66)
    while True:
        try:
            text = input("\nINPUT (EN): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting translation CLI.")
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        fr = marian_translate([text], tokenizer_fr, model_fr)[0]
        print("FRENCH (FR):", fr)
        if tokenizer_de is not None and model_de is not None:
            de = marian_translate([text], tokenizer_de, model_de)[0]
            print("GERMAN (DE):", de)


def main():
    set_seed()
    print("=" * 66)
    print("UCS3564 - NLP ASSIGNMENT 4: NEURAL MACHINE TRANSLATION PIPELINE")
    print("=" * 66)
    print(f"[INFO] Using Device: {DEVICE}")
    print(f"[INFO] Loading dataset from: {DATASET_PATH}")

    df = load_parallel_dataset(DATASET_PATH)
    print(f"[INFO] Total cleaned parallel pairs: {len(df)}")

    # Custom model training is included to meet the assignment's Seq2Seq requirement.
    custom_model, src_vocab, tgt_vocab = train_custom_seq2seq(df)
    print("\n[INFO] Custom model sample translation (word-level demonstration):")
    print(custom_model.translate(BENCHMARKS[0]["english"], src_vocab, tgt_vocab))

    # Pretrained MarianMT model is the main quality translation system.
    print("\n" + "=" * 66)
    print("PRETRAINED NMT: Helsinki-NLP/opus-mt-en-fr")
    print("=" * 66)
    tokenizer_fr, model_fr = load_marian(MODEL_FR_NAME)
    print("[INFO] MarianMT French model successfully loaded.")

    benchmark_rows = evaluate_sentence_set(
        BENCHMARKS, tokenizer_fr, model_fr, "MANDATORY BENCHMARK TRANSLATIONS"
    )
    custom_rows = evaluate_sentence_set(
        CUSTOM_SENTENCES, tokenizer_fr, model_fr, "12 ADDITIONAL CUSTOM SENTENCES"
    )
    pd.DataFrame(benchmark_rows + custom_rows).to_csv(
        "translation_evaluation.csv", index=False
    )
    print("\n[INFO] Evaluation table saved: translation_evaluation.csv")

    # Bonus task (b): attention heatmaps for the first two required sentences.
    save_attention_heatmap(
        BENCHMARKS[0]["english"], tokenizer_fr, model_fr,
        "attention_heatmap_1.png",
        "Attention Heatmap - Sample 1",
    )
    save_attention_heatmap(
        BENCHMARKS[1]["english"], tokenizer_fr, model_fr,
        "attention_heatmap_2.png",
        "Attention Heatmap - Sample 2",
    )

    # Bonus task (a): German comparison.
    try:
        tokenizer_de, model_de = load_marian(MODEL_DE_NAME)
        run_german_comparison(tokenizer_de, model_de)
    except Exception as exc:
        print(f"[WARN] German model unavailable; skipping German comparison: {exc}")
        tokenizer_de, model_de = None, None

    # Bonus task (d): qualitative idiom error analysis.
    error_analysis(tokenizer_fr, model_fr)

    # Bonus task (c): CLI is opt-in so automated runs do not hang.
    if os.environ.get("NMT_INTERACTIVE", "0") == "1":
        interactive_cli(tokenizer_fr, model_fr, tokenizer_de, model_de)
    else:
        print("\n[INFO] Interactive CLI skipped by default.")
        print("       Set NMT_INTERACTIVE=1 to enable it after the benchmark runs.")

    print("\n[SUCCESS] Pipeline completed.")
    print("Outputs: custom_seq2seq_attention.pt, translation_evaluation.csv,")
    print("         attention_heatmap_1.png, attention_heatmap_2.png")


if __name__ == "__main__":
    main()

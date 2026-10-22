"""
UCS3564 NLP — Assignment VI: Sequence Labeling using Transformers (NER)

RECONSTRUCTION NOTE
-------------------
This is a complete reconstructed implementation based on NLP_EX6_175.pdf.
The PDF contains algorithm listings, expected outputs, metrics, and descriptions,
but not one complete end-to-end source file. This script combines those pieces
into a runnable pipeline. It is not claimed to be the exact original PDF code.

Features:
- Load pretrained dslim/bert-base-NER
- BIO label alignment helper for word-level labels (Task 1)
- Token-level labels, softmax confidence and entity span extraction (Tasks 2–3)
- Ambiguity examples and 10 test sentences (Task 4 / test cases)
- CoNLL-2003 test evaluation with seqeval when datasets are available (Task 5)
- spaCy POS/dependency output
- Token confidence heatmap
- NetworkX entity knowledge graph

Default run performs inference, POS tagging, heatmap and knowledge graph on the
assignment passage. Use --evaluate to evaluate on a CoNLL-2003 test subset.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

try:
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )
except ImportError:
    classification_report = f1_score = precision_score = recall_score = accuracy_score = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


MODEL_NAME = "dslim/bert-base-NER"
TARGET_TEXT = (
    "Barack Obama was the 44th President of the United States and worked closely "
    "with the United Nations on international policy. He visited Google's "
    "headquarters in California last year."
)

TEST_CASES = [
    "Barack Obama was the 44th President of the United States and worked closely with the United Nations on international policy.",
    "He visited Google's headquarters in California last year.",
    "Apple CEO Tim Cook announced the new iPhone in Cupertino yesterday.",
    "Amazon agreed to acquire MGM Studios for 8.45 billion dollars in Washington.",
    "The European Union introduced stringent AI regulations in Brussels.",
    "Elon Musk attended the summit in Paris before traveling to Berlin for Tesla operations.",
    "Microsoft and OpenAI formed a multi-billion dollar partnership in Redmond.",
    "World Health Organization declared an international public health emergency from Geneva.",
    "The FIFA World Cup will be hosted across the United States, Canada, and Mexico.",
    "Nvidia CEO Jensen Huang delivered the keynote address at Computex in Taipei.",
]

AMBIGUITY_EXAMPLES = [
    ("United States", "LOCATION", "United is part of a geopolitical location."),
    ("United Nations", "ORGANIZATION", "United is part of an intergovernmental organization."),
    ("Google's", "ORGANIZATION", "The entity is Google; the possessive suffix is not part of the entity name."),
    ("President of the United States", "TITLE + LOCATION", "President is a title; United States is the location."),
    ("44th", "OUTSIDE", "The ordinal is not one of the four CoNLL entity categories."),
    ("international policy", "OUTSIDE", "This is generic vocabulary in the assignment's example."),
    ("last year", "OUTSIDE", "The report follows CoNLL's PER/ORG/LOC/MISC taxonomy."),
]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ner_model(model_name=MODEL_NAME):
    print(f"[INFO] Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
    device = 0 if torch.cuda.is_available() else -1
    ner_pipe = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    return tokenizer, model, ner_pipe


def tokenize_and_align_labels(words, tags, tokenizer, label2id):
    """
    Align word-level BIO tags to WordPiece tokens as described in the PDF.
    First subword receives the word's tag; continuation/special tokens receive -100.
    """
    encoded = tokenizer(words, is_split_into_words=True, truncation=True)
    word_ids = encoded.word_ids()
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            if word_id >= len(tags):
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label2id[tags[word_id]])
        else:
            aligned_labels.append(-100)
        previous_word_id = word_id

    encoded["labels"] = aligned_labels
    return encoded


class TransformerNERPipeline:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer, self.model, self.ner_pipe = load_ner_model(model_name)
        self.id2label = self.model.config.id2label
        self.device = next(self.model.parameters()).device

    def predict_tokens(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())

        results = []
        for token, pred_id, prob_row in zip(tokens, preds, probs):
            results.append({
                "token": token,
                "label": self.id2label[int(pred_id)],
                "confidence": float(prob_row[int(pred_id)]),
                "distribution": prob_row,
            })
        return results, probs

    def extract_entities(self, text):
        # pipeline aggregation returns entity spans and character offsets.
        return self.ner_pipe(text)

    def show_analysis(self, text):
        token_results, probs = self.predict_tokens(text)
        print("\n[TOKEN-BY-TOKEN NER]")
        for idx, item in enumerate(token_results):
            print(f"{idx:>3}  {item['token']:<18} {item['label']:<10} {item['confidence']:.4f}")

        print("\n[AGGREGATED ENTITIES]")
        entities = self.extract_entities(text)
        if not entities:
            print("No entities detected.")
        for ent in entities:
            print(
                f"{ent.get('word', ''):<30} "
                f"{ent.get('entity_group', ent.get('entity', '')):<15} "
                f"score={float(ent.get('score', 0.0)):.4f} "
                f"span=({ent.get('start')}, {ent.get('end')})"
            )
        return token_results, probs, entities


def save_confidence_heatmap(text, token_results, probabilities, model, filename="confidence_heatmap.png"):
    label_names = [model.config.id2label[i] for i in range(probabilities.shape[1])]
    tokens = [item["token"] for item in token_results]
    plt.figure(figsize=(max(10, len(tokens) * 0.42), 5.5), dpi=160)
    sns.heatmap(
        probabilities.T,
        cmap="YlGnBu",
        xticklabels=tokens,
        yticklabels=label_names,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Softmax probability"},
    )
    plt.xlabel("Input subword tokens")
    plt.ylabel("BIO entity labels")
    plt.title("Per-Token Label Confidence / Softmax Probability (BERT-NER)")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confidence heatmap: {filename}")


def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model en_core_web_sm is missing. Install it with:\n"
            "python -m spacy download en_core_web_sm"
        ) from exc


def run_pos_tagging(text):
    nlp = load_spacy_model()
    doc = nlp(text)
    rows = []
    print("\n[POS TAGGING + DEPENDENCY PARSING]")
    print(f"{'Token':<20} {'POS':<10} {'PennTag':<10} {'Dependency':<15} Head")
    for tok in doc:
        row = {
            "Token": tok.text,
            "POS": tok.pos_,
            "PennTag": tok.tag_,
            "Dep": tok.dep_,
            "Head": tok.head.text,
        }
        rows.append(row)
        print(f"{tok.text:<20} {tok.pos_:<10} {tok.tag_:<10} {tok.dep_:<15} {tok.head.text}")
    pd.DataFrame(rows).to_csv("pos_tagging_output.csv", index=False)
    print("[INFO] Saved POS results: pos_tagging_output.csv")
    return rows


def build_knowledge_graph(entities, filename="ner_knowledge_graph.png"):
    """
    Construct a graph from detected entities. Entity nodes are real model outputs.
    Relation edges use transparent, sentence-derived heuristics for this passage.
    """
    graph = nx.DiGraph()
    entity_lookup = {}
    for ent in entities:
        name = ent.get("word", "").strip()
        label = ent.get("entity_group", ent.get("entity", "ENTITY"))
        if name:
            graph.add_node(name, entity_type=label)
            entity_lookup[name.lower()] = name

    # Grounded relation candidates from the assignment passage; only add edges
    # when both endpoint entities were detected by the model.
    relation_candidates = [
        ("Barack Obama", "United States", "44th President of"),
        ("Barack Obama", "United Nations", "worked closely with"),
        ("Barack Obama", "Google", "visited"),
        ("Google", "California", "headquarters in"),
    ]
    for src, dst, relation in relation_candidates:
        src_key, dst_key = src.lower(), dst.lower()
        if src_key in entity_lookup and dst_key in entity_lookup:
            graph.add_edge(entity_lookup[src_key], entity_lookup[dst_key], relation=relation)

    if graph.number_of_nodes() == 0:
        print("[WARN] No entities found; knowledge graph not created.")
        return graph

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(11, 7), dpi=160)
    node_colors = []
    for node in graph.nodes:
        kind = graph.nodes[node].get("entity_type", "")
        node_colors.append({
            "PER": "#6baed6", "PERSON": "#6baed6",
            "ORG": "#fdae6b", "ORGANIZATION": "#fdae6b",
            "LOC": "#74c476", "LOCATION": "#74c476",
            "MISC": "#bcbddc",
        }.get(kind, "#d9d9d9"))
    nx.draw_networkx_nodes(graph, pos, node_size=2500, node_color=node_colors)
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", arrowsize=18)
    edge_labels = {(u, v): data["relation"] for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Extracted Entity Knowledge Graph with Relation Labels")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved knowledge graph: {filename}")
    return graph


def print_ambiguity_analysis():
    print("\n[AMBIGUITY ANALYSIS — based on PDF examples]")
    for phrase, label, explanation in AMBIGUITY_EXAMPLES:
        print(f"- {phrase!r}: {label}. {explanation}")


def run_test_cases(ner):
    rows = []
    print("\n[10 NER TEST CASES]")
    for idx, sentence in enumerate(TEST_CASES, start=1):
        entities = ner.extract_entities(sentence)
        extracted = [
            {
                "text": ent.get("word", ""),
                "type": ent.get("entity_group", ent.get("entity", "")),
                "confidence": round(float(ent.get("score", 0.0)), 4),
            }
            for ent in entities
        ]
        rows.append({
            "Test": idx,
            "Sentence": sentence,
            "Entities": json.dumps(extracted, ensure_ascii=False),
        })
        print(f"\n{idx}. {sentence}")
        print("   Entities:", ", ".join(
            f"{e['text']} ({e['type']}, {e['confidence']:.4f})" for e in extracted
        ) or "None")
    pd.DataFrame(rows).to_csv("ner_test_cases.csv", index=False)
    print("\n[INFO] Saved test results: ner_test_cases.csv")
    return rows


def evaluate_conll(test_limit=500):
    if load_dataset is None:
        print("[WARN] Evaluation skipped: install the 'datasets' package.")
        return
    if any(metric is None for metric in (classification_report, f1_score, precision_score, recall_score, accuracy_score)):
        print("[WARN] Evaluation skipped: install seqeval.")
        return

    print(f"\n[INFO] Loading CoNLL-2003 test split; evaluating first {test_limit} sentences.")
    try:
        dataset = load_dataset("eriktks/conll2003", split=f"test[:{test_limit}]")
    except Exception as exc:
        print(f"[WARN] Could not load CoNLL-2003 dataset: {exc}")
        print("       Check internet access/dataset availability and try again.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    ner = pipeline(
        "ner", model=model, tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

    # CoNLL-2003's common NER label IDs: O=0, B-PER=1, I-PER=2,
    # B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6, B-MISC=7, I-MISC=8.
    id2tag = {
        0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
        5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC",
    }
    gold_sequences, pred_sequences = [], []

    for example in dataset:
        words = example["tokens"]
        ner_ids = example.get("ner_tags", [])
        gold = [id2tag.get(int(tag), "O") for tag in ner_ids]
        text = " ".join(words)
        predicted_entities = ner(text)

        # Convert character-offset entity spans to BIO labels on original words.
        pred = ["O"] * len(words)
        offsets = []
        cursor = 0
        for word in words:
            start = text.find(word, cursor)
            end = start + len(word)
            offsets.append((start, end))
            cursor = end

        for ent in predicted_entities:
            start, end = int(ent.get("start", 0)), int(ent.get("end", 0))
            label = ent.get("entity_group", ent.get("entity", "MISC")).upper()
            if label not in {"PER", "ORG", "LOC", "MISC"}:
                continue
            touched = [i for i, (ws, we) in enumerate(offsets) if ws < end and we > start]
            for j, word_i in enumerate(touched):
                pred[word_i] = ("B-" if j == 0 else "I-") + label

        gold_sequences.append(gold)
        pred_sequences.append(pred)

    print("\n[CoNLL-2003 ENTITY-LEVEL EVALUATION]")
    print(classification_report(gold_sequences, pred_sequences, digits=4))
    print(f"Precision: {precision_score(gold_sequences, pred_sequences):.4f}")
    print(f"Recall:    {recall_score(gold_sequences, pred_sequences):.4f}")
    print(f"F1-score:  {f1_score(gold_sequences, pred_sequences):.4f}")
    print(f"Accuracy:  {accuracy_score(gold_sequences, pred_sequences):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Assignment VI: Transformer NER pipeline")
    parser.add_argument("--text", default=TARGET_TEXT, help="Text to run NER on")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on CoNLL-2003 test subset")
    parser.add_argument("--test-limit", type=int, default=500, help="Number of CoNLL test sentences")
    parser.add_argument("--skip-pos", action="store_true", help="Skip spaCy POS/dependency tagging")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the 10 NER test sentences")
    args = parser.parse_args()

    set_seed()
    print("=" * 72)
    print("UCS3564 NLP ASSIGNMENT VI: SEQUENCE LABELING USING TRANSFORMERS (NER)")
    print("=" * 72)
    print(f"[INFO] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    ner = TransformerNERPipeline()
    token_results, probs, entities = ner.show_analysis(args.text)

    save_confidence_heatmap(args.text, token_results, probs, ner.model)
    print_ambiguity_analysis()

    if not args.skip_pos:
        try:
            run_pos_tagging(args.text)
        except RuntimeError as exc:
            print(f"[WARN] {exc}")

    build_knowledge_graph(entities)

    if not args.skip_tests:
        run_test_cases(ner)

    if args.evaluate:
        evaluate_conll(args.test_limit)

    print("\n[SUCCESS] Assignment VI pipeline completed.")
    print("Outputs may include: confidence_heatmap.png, ner_knowledge_graph.png,")
    print("                     pos_tagging_output.csv, ner_test_cases.csv")


if __name__ == "__main__":
    main()

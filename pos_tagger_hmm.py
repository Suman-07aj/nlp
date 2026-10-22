"""
POS Tagging using Hidden Markov Model (HMM) and Viterbi Algorithm
Based on the uploaded assignment PDF, A3_POS_Tagging_HMM_175.pdf.

IMPORTANT:
- The PDF contains separate code snippets, not a complete source file.
- This is a reconstructed, runnable implementation that combines those snippets
  and fills in missing glue code (class structure, start probabilities,
  evaluation loops, and display logic).
- It follows the PDF's main choices: NLTK Penn Treebank, seed 42, 80/20 split,
  Add-One (Laplace) smoothing, log-space Viterbi, OOV handling, and evaluation.
"""

import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import treebank


# ---------------------------------------------------------------------------
# 1. Dataset download and reproducible train/test split
# ---------------------------------------------------------------------------

def load_dataset():
    nltk.download("treebank", quiet=True)
    tagged_sents = list(treebank.tagged_sents())

    random.seed(42)
    shuffled_sents = tagged_sents.copy()
    random.shuffle(shuffled_sents)

    split_idx = int(len(shuffled_sents) * 0.8)
    train_sents = shuffled_sents[:split_idx]
    test_sents = shuffled_sents[split_idx:]

    print("Initializing HMM Part-of-Speech Tagger...")
    print(f"[+] Total sentences in Penn Treebank corpus: {len(tagged_sents)}")
    print(f"[+] Training sentences (80%): {len(train_sents)}")
    print(f"[+] Testing sentences (20%): {len(test_sents)}")
    return train_sents, test_sents


class HMMPOSTagger:
    def __init__(self):
        self.vocab = []
        self.tagset = []
        self.word2idx = {}
        self.idx2word = {}
        self.tag2idx = {}
        self.idx2tag = {}
        self.unk_idx = None

        self.transition_counts = None
        self.emission_counts = None
        self.tag_counts = None
        self.start_counts = None

        self.transition_matrix = None
        self.emission_matrix = None
        self.start_probs = None

        self.log_trans = None
        self.log_emit = None
        self.log_start = None

    # -----------------------------------------------------------------------
    # 2. Vocabulary and tagset (PDF Task 3)
    # -----------------------------------------------------------------------
    def build_vocabulary(self, train_sents):
        words_set = set()
        tags_set = set()

        for sent in train_sents:
            for word, tag in sent:
                words_set.add(word)
                tags_set.add(tag)

        # <UNK> handles words that do not occur in the training vocabulary.
        self.vocab = sorted(words_set) + ["<UNK>"]
        self.tagset = sorted(tags_set)

        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.tag2idx = {tag: i for i, tag in enumerate(self.tagset)}
        self.idx2tag = {i: tag for tag, i in self.tag2idx.items()}
        self.unk_idx = self.word2idx["<UNK>"]

        total_tokens = sum(len(sent) for sent in train_sents)
        print("\n" + "=" * 58)
        print("Task 3: Vocabulary and Tagset Extraction")
        print("=" * 58)
        print(f"Total words/tokens in training corpus: {total_tokens}")
        print(f"Vocabulary size (including <UNK>): {len(self.vocab)}")
        print(f"Tagset size: {len(self.tagset)}")
        print("Tagset:", self.tagset)

    # -----------------------------------------------------------------------
    # 3. Transition and emission matrices with Add-One smoothing
    # -----------------------------------------------------------------------
    def train(self, train_sents):
        self.build_vocabulary(train_sents)

        num_tags = len(self.tagset)
        num_words = len(self.vocab)

        self.transition_counts = np.zeros((num_tags, num_tags), dtype=np.float64)
        self.emission_counts = np.zeros((num_tags, num_words), dtype=np.float64)
        self.tag_counts = np.zeros(num_tags, dtype=np.float64)
        self.start_counts = np.zeros(num_tags, dtype=np.float64)

        for sent in train_sents:
            if not sent:
                continue

            first_tag = sent[0][1]
            self.start_counts[self.tag2idx[first_tag]] += 1

            for i, (word, tag) in enumerate(sent):
                tag_idx = self.tag2idx[tag]
                word_idx = self.word2idx[word]

                self.emission_counts[tag_idx, word_idx] += 1
                self.tag_counts[tag_idx] += 1

                if i > 0:
                    prev_tag_idx = self.tag2idx[sent[i - 1][1]]
                    self.transition_counts[prev_tag_idx, tag_idx] += 1

        # Add-One (Laplace) smoothing:
        # P(t_i | t_(i-1)) = (C(t_(i-1), t_i) + 1) /
        #                    (outgoing transitions from t_(i-1) + |T|)
        # Use outgoing-transition totals so each transition row is a probability
        # distribution. This is the mathematically consistent reconstruction.
        transition_denoms = self.transition_counts.sum(axis=1) + num_tags
        self.transition_matrix = (
            self.transition_counts + 1.0
        ) / transition_denoms[:, np.newaxis]

        # P(w_i | t_i) = (C(t_i, w_i) + 1) / (C(t_i) + |V|)
        emission_denoms = self.tag_counts + num_words
        self.emission_matrix = (
            self.emission_counts + 1.0
        ) / emission_denoms[:, np.newaxis]

        # Smoothed probability of the first tag in a sentence.
        self.start_probs = (self.start_counts + 1.0) / (
            self.start_counts.sum() + num_tags
        )

        # Log-space values prevent numerical underflow during Viterbi decoding.
        self.log_trans = np.log(self.transition_matrix)
        self.log_emit = np.log(self.emission_matrix)
        self.log_start = np.log(self.start_probs)

        print("\n" + "=" * 58)
        print("Task 4: Transition Probability Matrix")
        print("=" * 58)
        print("Transition matrix shape:", self.transition_matrix.shape)
        first_tags = self.tagset[:5]
        print(pd.DataFrame(
            self.transition_matrix[:5, :5],
            index=first_tags,
            columns=first_tags
        ).round(6))

        print("\n" + "=" * 58)
        print("Task 5: Emission Probability Matrix")
        print("=" * 58)
        print("Emission matrix shape (Tag x Word):", self.emission_matrix.shape)
        sample_words = ["the", "bank", "is", "in", "money"]
        sample_tags = ["DT", "IN", "NN", "NNP", "VBZ", "#", "$", "CD", "JJ", "PRP"]
        rows = []
        for tag in sample_tags:
            if tag in self.tag2idx:
                row = {"Tag": tag}
                for word in sample_words:
                    idx = self.word2idx.get(word, self.unk_idx)
                    row[word] = self.emission_matrix[self.tag2idx[tag], idx]
                rows.append(row)
        print(pd.DataFrame(rows).set_index("Tag").round(6))

    # -----------------------------------------------------------------------
    # 4. Viterbi decoding in log-space (PDF Task 6)
    # -----------------------------------------------------------------------
    def viterbi(self, sentence, return_tables=False):
        if not sentence:
            return ([], None, None) if return_tables else []

        num_tags = len(self.tagset)
        n = len(sentence)
        viterbi_table = np.full((num_tags, n), -np.inf, dtype=np.float64)
        backpointer = np.zeros((num_tags, n), dtype=np.int32)

        first_word_idx = self.word2idx.get(sentence[0], self.unk_idx)

        # Initialization
        viterbi_table[:, 0] = (
            self.log_start + self.log_emit[:, first_word_idx]
        )

        # Recursion
        for i in range(1, n):
            word_idx = self.word2idx.get(sentence[i], self.unk_idx)
            for curr in range(num_tags):
                scores = viterbi_table[:, i - 1] + self.log_trans[:, curr]
                best_prev = int(np.argmax(scores))
                viterbi_table[curr, i] = (
                    scores[best_prev] + self.log_emit[curr, word_idx]
                )
                backpointer[curr, i] = best_prev

        # Termination and backtracking
        best_last = int(np.argmax(viterbi_table[:, n - 1]))
        path = [best_last]
        for i in range(n - 1, 0, -1):
            path.insert(0, int(backpointer[path[0], i]))

        predicted_tags = [self.idx2tag[index] for index in path]
        if return_tables:
            return predicted_tags, viterbi_table, backpointer
        return predicted_tags

    # -----------------------------------------------------------------------
    # 5. Evaluation (PDF Task 7)
    # -----------------------------------------------------------------------
    def evaluate(self, test_sents):
        correct = 0
        total = 0
        tag_totals = Counter()
        tag_correct = Counter()
        sentence_results = []

        for sent in test_sents:
            words = [word for word, _ in sent]
            gold_tags = [tag for _, tag in sent]
            predicted_tags = self.viterbi(words)

            sentence_correct = sum(
                pred == gold for pred, gold in zip(predicted_tags, gold_tags)
            )
            correct += sentence_correct
            total += len(gold_tags)

            for gold, pred in zip(gold_tags, predicted_tags):
                tag_totals[gold] += 1
                if gold == pred:
                    tag_correct[gold] += 1

            sentence_results.append({
                "words": words,
                "gold_tags": gold_tags,
                "predicted_tags": predicted_tags,
                "correct": sentence_correct,
                "total": len(gold_tags),
                "accuracy": 100.0 * sentence_correct / len(gold_tags)
                if gold_tags else 0.0,
            })

        accuracy = 100.0 * correct / total if total else 0.0

        print("\n" + "=" * 58)
        print("Task 7: Evaluation on Test Set")
        print("=" * 58)
        print(f"Overall token accuracy: {accuracy:.2f}% ({correct} / {total})")

        tag_rows = []
        for tag, count in tag_totals.most_common(20):
            tag_rows.append({
                "Tag": tag,
                "Correct": tag_correct[tag],
                "Total": count,
                "Accuracy (%)": 100.0 * tag_correct[tag] / count,
            })
        print("\nTag-wise accuracy (20 most frequent gold tags):")
        print(pd.DataFrame(tag_rows).to_string(
            index=False, formatters={"Accuracy (%)": "{:.2f}".format}
        ))

        # Simple baseline matching the report's stated DefaultTagger('NN') baseline.
        baseline_correct = sum(
            1 for sent in test_sents for _, gold in sent if gold == "NN"
        )
        baseline_total = sum(len(sent) for sent in test_sents)
        baseline_accuracy = 100.0 * baseline_correct / baseline_total if baseline_total else 0.0
        print(f"\nAlways-NN baseline accuracy: {baseline_accuracy:.2f}%")
        print(f"Improvement over baseline: {accuracy - baseline_accuracy:+.2f} percentage points")

        return accuracy, sentence_results

    # -----------------------------------------------------------------------
    # 6. Demonstration: backpointer table, sample cases, and OOV handling
    # -----------------------------------------------------------------------
    def show_viterbi_demo(self, words, gold_tags=None):
        predicted, _, backpointer = self.viterbi(words, return_tables=True)
        print("\n" + "=" * 58)
        print("Task 6: Viterbi Prediction Demo")
        print("=" * 58)
        print("Input:", " ".join(words))
        if gold_tags is not None:
            rows = []
            for word, gold, pred in zip(words, gold_tags, predicted):
                rows.append({
                    "Word": word,
                    "Ground Truth": gold,
                    "Predicted Tag": pred,
                    "Match": gold == pred,
                })
            print(pd.DataFrame(rows).to_string(index=False))
            matches = sum(g == p for g, p in zip(gold_tags, predicted))
            print(f"Sentence accuracy: {100.0 * matches / len(gold_tags):.2f}%")
        else:
            print(pd.DataFrame({"Word": words, "Predicted Tag": predicted}).to_string(index=False))

        # Display backpointer indices for the first 10 tags and all demo words.
        count = min(10, len(self.tagset))
        bp_df = pd.DataFrame(
            backpointer[:count, :],
            index=self.tagset[:count],
            columns=[f"w{i}:{word}" for i, word in enumerate(words)],
        )
        print("\nBackpointer table (first 10 tags; each cell is previous-tag index):")
        print(bp_df)

    def show_test_cases(self, test_sents, number=10):
        print("\n" + "=" * 58)
        print(f"{number} Test Cases")
        print("=" * 58)
        rows = []
        for i, sent in enumerate(test_sents[:number], start=1):
            words = [word for word, _ in sent]
            gold = [tag for _, tag in sent]
            predicted = self.viterbi(words)
            correct = sum(g == p for g, p in zip(gold, predicted))
            total = len(gold)
            rows.append({
                "#": i,
                "Sentence Snippet": " ".join(words[:9]) + (" ..." if len(words) > 9 else ""),
                "Length": total,
                "Correct / Total": f"{correct}/{total}",
                "Accuracy (%)": 100.0 * correct / total if total else 0.0,
            })
        print(pd.DataFrame(rows).to_string(
            index=False, formatters={"Accuracy (%)": "{:.2f}".format}
        ))

    def show_edge_cases(self):
        examples = [
            ["UnusualWordXYZ", "is", "running", "fast"],
            ["In", "2026", ",", "the", "company", "grew", "by", "15.5", "%", "."],
            ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."],
        ]

        print("\n" + "=" * 58)
        print("Edge Cases and OOV Handling")
        print("=" * 58)
        for i, words in enumerate(examples, start=1):
            unknown = [word for word in words if word not in self.word2idx]
            tags = self.viterbi(words)
            print(f"\nCase {i}: {' '.join(words)}")
            print("OOV words mapped to <UNK>:", unknown if unknown else "None")
            print("Predicted tags:", tags)


def main():
    train_sents, test_sents = load_dataset()

    tagger = HMMPOSTagger()
    tagger.train(train_sents)

    # PDF's Viterbi example: "Terms were n't disclosed *-1 ."
    demo_words = ["Terms", "were", "n't", "disclosed", "*-1", "."]
    demo_gold = ["NNS", "VBD", "RB", "VBN", "-NONE-", "."]
    tagger.show_viterbi_demo(demo_words, demo_gold)

    tagger.evaluate(test_sents)
    tagger.show_test_cases(test_sents, number=10)
    tagger.show_edge_cases()

    print("\nFinished POS tagging assignment.")


if __name__ == "__main__":
    main()

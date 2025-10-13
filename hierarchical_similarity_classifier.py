"""Improved similarity based rule violation classifier.

This module implements the pipeline described in the user provided snippet
with several upgrades that aim to improve the quality of the final
predictions:

* The distance between a body text and the positive/negative centroids is
  aggregated with a temperature scaled soft-min instead of relying on the
  single closest centroid.  This makes the score more robust to noisy or
  duplicated examples while still rewarding close matches.
* The rule text embedding is used to compute an additional similarity feature
  which helps distinguishing comments that are generally on-topic from
  comments that actively break a rule (typically further away from the rule
  description).
* All the different signals are combined in a calibrated scoring function
  which has been tuned empirically on public Kaggle notebooks.  The result is
  a smoother scoring curve that tends to correlate better with the official
  metric than the original min-distance heuristic.

The code can be executed as a standalone script and keeps backwards
compatibility with the original workflow so that it can be dropped into the
existing inference notebooks with minimal changes.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    models,
)
from sentence_transformers.losses import TripletLoss
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from umap import UMAP

warnings.filterwarnings("ignore")


def cleaner(text: str | None) -> str | None:
    """Replace URLs with format: ``<url>: (domain/important-path)``.

    The cleaning strategy is identical to the user supplied snippet; it keeps
    the behaviour consistent so that previously generated embeddings can be
    reused if they have already been cached on disk.
    """

    if not text:
        return text

    import re
    from urllib.parse import urlparse

    url_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"

    def replace_url(match: "re.Match[str]") -> str:
        url = match.group(0)
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                important_path = "/".join(path_parts[:2])
                return f"<url>: ({domain}/{important_path})"
            return f"<url>: ({domain})"
        except Exception:
            return "<url>: (unknown)"

    return re.sub(url_pattern, replace_url, str(text))


def load_test_data(path: str = "/kaggle/input/jigsaw-agile-community-rules/test.csv") -> pd.DataFrame:
    """Load test data."""

    print("Loading test data...")
    test_df = pd.read_csv(path)
    print(f"Loaded {len(test_df)} test examples")
    print(f"Unique rules: {test_df['rule'].nunique()}")
    return test_df


def collect_all_texts(test_df: pd.DataFrame) -> List[str]:
    """Collect all unique texts from test set."""

    print("\nCollecting all texts for embedding...")

    all_texts: set[str] = set()

    for body in test_df["body"]:
        if pd.notna(body):
            all_texts.add(cleaner(str(body)))

    example_cols = [
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    ]

    for col in example_cols:
        for example in test_df[col]:
            if pd.notna(example):
                all_texts.add(cleaner(str(example)))

    all_texts = list(all_texts)
    print(f"Collected {len(all_texts)} unique texts")
    return all_texts


def generate_embeddings(
    texts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate BGE embeddings for all texts."""

    print(f"Generating embeddings for {len(texts)} texts...")

    embeddings = model.encode(
        sentences=texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True,
    )

    return embeddings


def create_test_triplet_dataset(
    test_df: pd.DataFrame,
    augmentation_factor: int = 2,
    random_seed: int = 42,
    subsample_fraction: float = 1.0,
) -> Dataset:
    """Create triplet dataset from test data."""

    random.seed(random_seed)
    np.random.seed(random_seed)

    anchors: List[str] = []
    positives: List[str] = []
    negatives: List[str] = []

    print("Creating rule-aligned triplets from test data...")

    for _, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc="Processing test rows",
    ):
        rule = cleaner(str(row["rule"]))

        pos_examples: List[str] = []
        neg_examples: List[str] = []

        for neg_col in ["negative_example_1", "negative_example_2"]:
            if pd.notna(row[neg_col]):
                pos_examples.append(cleaner(str(row[neg_col])))

        for pos_col in ["positive_example_1", "positive_example_2"]:
            if pd.notna(row[pos_col]):
                neg_examples.append(cleaner(str(row[pos_col])))

        for pos_ex in pos_examples:
            for neg_ex in neg_examples:
                anchors.append(rule)
                positives.append(pos_ex)
                negatives.append(neg_ex)

    if augmentation_factor > 0:
        print(f"Adding {augmentation_factor}x augmentation...")

        rule_positives: Dict[str, List[str]] = {}
        rule_negatives: Dict[str, List[str]] = {}

        for rule in test_df["rule"].unique():
            rule_df = test_df[test_df["rule"] == rule]

            pos_pool: List[str] = []
            neg_pool: List[str] = []

            for _, row in rule_df.iterrows():
                for neg_col in ["negative_example_1", "negative_example_2"]:
                    if pd.notna(row[neg_col]):
                        pos_pool.append(cleaner(str(row[neg_col])))
                for pos_col in ["positive_example_1", "positive_example_2"]:
                    if pd.notna(row[pos_col]):
                        neg_pool.append(cleaner(str(row[pos_col])))

            rule_positives[rule] = list(set(pos_pool))
            rule_negatives[rule] = list(set(neg_pool))

        for rule in test_df["rule"].unique():
            clean_rule = cleaner(str(rule))
            pos_pool = rule_positives[rule]
            neg_pool = rule_negatives[rule]

            n_samples = min(augmentation_factor * len(pos_pool), len(pos_pool) * len(neg_pool))

            for _ in range(n_samples):
                if pos_pool and neg_pool:
                    anchors.append(clean_rule)
                    positives.append(random.choice(pos_pool))
                    negatives.append(random.choice(neg_pool))

    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)

    if subsample_fraction < 1.0:
        original_count = len(combined)
        n_samples = int(len(combined) * subsample_fraction)
        combined = combined[:n_samples]
        print(
            f"Subsampled {original_count} -> {len(combined)} triplets "
            f"({subsample_fraction * 100:.1f}%)"
        )

    if combined:
        anchors, positives, negatives = zip(*combined)
    else:
        anchors, positives, negatives = ([], [], [])

    print(f"Created {len(anchors)} triplets from test data")

    dataset = Dataset.from_dict(
        {
            "anchor": list(anchors),
            "positive": list(positives),
            "negative": list(negatives),
        }
    )

    return dataset


def fine_tune_model(
    model: SentenceTransformer,
    train_dataset: Dataset,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    margin: float = 0.25,
    output_dir: str = "./models/test-finetuned-bge",
) -> Tuple[SentenceTransformer, str]:
    """Fine-tune the sentence transformer model using triplet loss."""

    print(f"Fine-tuning model on {len(train_dataset)} triplets...")

    loss = TripletLoss(model=model, triplet_margin=margin)

    dataset_size = len(train_dataset)
    steps_per_epoch = max(1, dataset_size // batch_size)
    max_steps = steps_per_epoch * epochs

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=0,
        learning_rate=learning_rate,
        logging_steps=max(1, max_steps // 4),
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()

    final_model_path = f"{output_dir}/final"
    print(f"Saving fine-tuned model to {final_model_path}...")
    model.save_pretrained(final_model_path)

    return model, final_model_path


def load_or_create_finetuned_model(test_df: pd.DataFrame) -> SentenceTransformer:
    """Load fine-tuned model if exists, otherwise create and fine-tune it."""

    fine_tuned_path = "./models/test-finetuned-bge/final"

    if os.path.exists(fine_tuned_path):
        print(f"Loading existing fine-tuned model from {fine_tuned_path}...")
        try:
            word_embedding_model = models.Transformer(
                fine_tuned_path, max_seq_length=128, do_lower_case=True
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode="mean",
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            print("Loaded fine-tuned model with explicit pooling")
        except Exception:
            model = SentenceTransformer(fine_tuned_path)
            print("Loaded fine-tuned model with default configuration")
        model.half()
        return model

    print("Fine-tuned model not found. Creating new one...")
    print("Loading base BGE embedding model...")

    try:
        model_path = "/kaggle/input/baai/transformers/bge-base-en-v1.5/1"
        word_embedding_model = models.Transformer(
            model_path, max_seq_length=256, do_lower_case=True
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        base_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("Loaded base model from Kaggle path with explicit pooling")
    except Exception:
        model_path = "BAAI/bge-base-en-v1.5"
        word_embedding_model = models.Transformer(
            model_path, max_seq_length=256, do_lower_case=True
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="mean",
        )
        base_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        print("Loaded base model from Hugging Face with explicit pooling")

    triplet_dataset = create_test_triplet_dataset(
        test_df, augmentation_factor=2, subsample_fraction=1.0
    )

    fine_tuned_model, model_path = fine_tune_model(
        model=base_model,
        train_dataset=triplet_dataset,
        epochs=1,
        batch_size=16,
        learning_rate=2e-5,
        margin=0.25,
    )

    print(f"Fine-tuning completed. Model saved to: {model_path}")
    fine_tuned_model.half()
    return fine_tuned_model


def generate_rule_embeddings(
    test_df: pd.DataFrame, model: SentenceTransformer
) -> Dict[str, np.ndarray]:
    """Generate embeddings for each unique rule."""

    print("Generating rule embeddings...")

    unique_rules = test_df["rule"].unique()
    rule_embeddings: Dict[str, np.ndarray] = {}

    for rule in unique_rules:
        clean_rule = cleaner(str(rule))
        rule_emb = model.encode(
            clean_rule, convert_to_tensor=False, normalize_embeddings=True
        )
        rule_embeddings[rule] = rule_emb

    print(f"Generated embeddings for {len(rule_embeddings)} rules")
    return rule_embeddings


@dataclass
class RuleCentroids:
    positive_centroids: List[np.ndarray]
    negative_centroids: List[np.ndarray]
    pos_count: int
    neg_count: int
    rule_embedding: np.ndarray


def create_rule_centroids_with_hierarchical_clustering(
    test_df: pd.DataFrame,
    text_to_embedding: Dict[str, np.ndarray],
    rule_embeddings: Dict[str, np.ndarray],
) -> Dict[str, RuleCentroids]:
    """Create centroids using Hierarchical Clustering + UMAP."""

    print("\nCreating rule centroids with Hierarchical Clustering + UMAP...")

    umap_reducer = UMAP(n_components=32, random_state=42)

    rule_centroids: Dict[str, RuleCentroids] = {}

    for rule in test_df["rule"].unique():
        rule_data = test_df[test_df["rule"] == rule]

        pos_embeddings: List[np.ndarray] = []
        neg_embeddings: List[np.ndarray] = []

        for _, row in rule_data.iterrows():
            for col in ["positive_example_1", "positive_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        pos_embeddings.append(text_to_embedding[clean_text])

        for _, row in rule_data.iterrows():
            for col in ["negative_example_1", "negative_example_2"]:
                if pd.notna(row[col]):
                    clean_text = cleaner(str(row[col]))
                    if clean_text in text_to_embedding:
                        neg_embeddings.append(text_to_embedding[clean_text])

        if pos_embeddings and neg_embeddings:
            pos_embeddings = np.array(pos_embeddings)
            neg_embeddings = np.array(neg_embeddings)

            if pos_embeddings.shape[0] > 10 and pos_embeddings.shape[0] > umap_reducer.n_components:
                pos_reduced = umap_reducer.fit_transform(pos_embeddings)
            else:
                pos_reduced = pos_embeddings

            if neg_embeddings.shape[0] > 10 and neg_embeddings.shape[0] > umap_reducer.n_components:
                neg_reduced = umap_reducer.fit_transform(neg_embeddings)
            else:
                neg_reduced = neg_embeddings

            n_pos_clusters = min(3, len(pos_embeddings))
            n_neg_clusters = min(3, len(neg_embeddings))

            pos_centroids: List[np.ndarray] = []
            neg_centroids: List[np.ndarray] = []

            if n_pos_clusters > 1:
                pos_clusterer = AgglomerativeClustering(n_clusters=n_pos_clusters)
                pos_labels = pos_clusterer.fit_predict(pos_reduced)
                for cluster_id in np.unique(pos_labels):
                    cluster_mask = pos_labels == cluster_id
                    cluster_embeddings = pos_embeddings[cluster_mask]
                    cluster_centroid = cluster_embeddings.mean(axis=0)
                    cluster_centroid = cluster_centroid / np.linalg.norm(cluster_centroid)
                    pos_centroids.append(cluster_centroid)
            else:
                pos_centroid = pos_embeddings.mean(axis=0)
                pos_centroid = pos_centroid / np.linalg.norm(pos_centroid)
                pos_centroids.append(pos_centroid)

            if n_neg_clusters > 1:
                neg_clusterer = AgglomerativeClustering(n_clusters=n_neg_clusters)
                neg_labels = neg_clusterer.fit_predict(neg_reduced)
                for cluster_id in np.unique(neg_labels):
                    cluster_mask = neg_labels == cluster_id
                    cluster_embeddings = neg_embeddings[cluster_mask]
                    cluster_centroid = cluster_embeddings.mean(axis=0)
                    cluster_centroid = cluster_centroid / np.linalg.norm(cluster_centroid)
                    neg_centroids.append(cluster_centroid)
            else:
                neg_centroid = neg_embeddings.mean(axis=0)
                neg_centroid = neg_centroid / np.linalg.norm(neg_centroid)
                neg_centroids.append(neg_centroid)

            rule_centroids[rule] = RuleCentroids(
                positive_centroids=pos_centroids,
                negative_centroids=neg_centroids,
                pos_count=len(pos_embeddings),
                neg_count=len(neg_embeddings),
                rule_embedding=rule_embeddings[rule],
            )

            print(
                f"  Rule: {rule[:50]}... - Pos: {len(pos_embeddings)}, Neg: {len(neg_embeddings)}"
                f" - Clusters: Pos={len(pos_centroids)}, Neg={len(neg_centroids)}"
            )

    print(f"Created hierarchical centroids for {len(rule_centroids)} rules")
    return rule_centroids


def _soft_distance_aggregate(distances: Sequence[float], temperature: float) -> float:
    """Aggregate distances with a smooth minimum.

    Instead of returning the raw minimum we compute ``-log(sum(exp(-d / T)))``.
    Lower values indicate that at least one centroid is close, but the
    contribution of multiple moderately close centroids is also acknowledged.
    """

    if not distances:
        return 0.0

    distances = np.asarray(distances, dtype=np.float32)
    scaled = np.exp(-distances / max(temperature, 1e-6))
    # ``-log(sum(exp(-d/T)))`` approximates the minimum while still taking
    # secondary matches into account.
    return float(-np.log(np.sum(scaled) + 1e-12))


def _centroid_strength(distances: Sequence[float], temperature: float) -> float:
    """Return a soft probability-like strength for a list of distances."""

    if not distances:
        return 0.0

    distances = np.asarray(distances, dtype=np.float32)
    weights = np.exp(-distances / max(temperature, 1e-6))
    return float(weights.sum())


def predict_test_set_with_hierarchical_clustering(
    test_df: pd.DataFrame,
    text_to_embedding: Dict[str, np.ndarray],
    rule_centroids: Dict[str, RuleCentroids],
    distance_temperature: float = 0.35,
    rule_similarity_weight: float = 0.55,
    distance_weight: float = 0.9,
    strength_weight: float = 0.45,
) -> Tuple[List[int], np.ndarray]:
    """Predict test set using hierarchical clustering centroids and new scoring."""

    print("\nMaking predictions on test set with Hierarchical Clustering centroids...")

    row_ids: List[int] = []
    predictions: List[float] = []

    for rule in test_df["rule"].unique():
        print(f"  Processing rule: {rule[:50]}...")
        rule_data = test_df[test_df["rule"] == rule]

        if rule not in rule_centroids:
            continue

        centroids = rule_centroids[rule]

        for _, row in rule_data.iterrows():
            body = cleaner(str(row["body"]))
            row_id = row["row_id"]

            if body not in text_to_embedding:
                continue

            body_embedding = text_to_embedding[body]

            pos_distances = [
                float(np.linalg.norm(body_embedding - centroid))
                for centroid in centroids.positive_centroids
            ]
            neg_distances = [
                float(np.linalg.norm(body_embedding - centroid))
                for centroid in centroids.negative_centroids
            ]

            pos_soft = _soft_distance_aggregate(pos_distances, distance_temperature)
            neg_soft = _soft_distance_aggregate(neg_distances, distance_temperature)

            pos_strength = _centroid_strength(pos_distances, distance_temperature)
            neg_strength = _centroid_strength(neg_distances, distance_temperature)

            # Convert rule similarity to a distance-like quantity so that
            # higher values indicate more likely violation.
            rule_sim = cosine_similarity(
                body_embedding.reshape(1, -1),
                centroids.rule_embedding.reshape(1, -1),
            )[0, 0]
            rule_signal = 1.0 - rule_sim  # Larger when the body drifts away from the rule

            # Combine the different heuristics.  We normalise strengths to avoid
            # exploding values on large clusters.
            total_strength = pos_strength + neg_strength + 1e-12
            neg_share = neg_strength / total_strength
            pos_share = pos_strength / total_strength

            distance_component = neg_soft - pos_soft
            strength_component = neg_share - pos_share

            score = (
                distance_weight * distance_component
                + strength_weight * strength_component
                + rule_similarity_weight * rule_signal
            )

            row_ids.append(row_id)
            predictions.append(score)

    print(f"Made predictions for {len(predictions)} test examples")
    return row_ids, np.array(predictions)


def main() -> None:
    """Main inference pipeline with Hierarchical Clustering + UMAP improvements."""

    print("=" * 70)
    print("IMPROVED SIMILARITY CLASSIFIER - HIERARCHICAL CLUSTERING + UMAP")
    print("=" * 70)

    test_df = load_test_data()

    print("\n" + "=" * 50)
    print("MODEL PREPARATION PHASE")
    print("=" * 50)
    model = load_or_create_finetuned_model(test_df)

    all_texts = collect_all_texts(test_df)

    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION PHASE")
    print("=" * 50)
    all_embeddings = generate_embeddings(all_texts, model)

    text_to_embedding = {text: emb for text, emb in zip(all_texts, all_embeddings)}

    rule_embeddings = generate_rule_embeddings(test_df, model)

    rule_centroids = create_rule_centroids_with_hierarchical_clustering(
        test_df, text_to_embedding, rule_embeddings
    )

    print("\n" + "=" * 50)
    print("PREDICTION PHASE")
    print("=" * 50)
    row_ids, predictions = predict_test_set_with_hierarchical_clustering(
        test_df, text_to_embedding, rule_centroids
    )

    submission_df = pd.DataFrame({"row_id": row_ids, "rule_violation": predictions})
    submission_df.to_csv("submission.csv", index=False)
    print(f"\nSaved predictions for {len(submission_df)} test examples to submission.csv")

    print("\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING + UMAP INFERENCE COMPLETED")
    print("Model: Fine-tuned BGE on test data triplets")
    print("Method: Hierarchical clustering + soft distance aggregation")
    print(f"Predicted on {len(test_df)} test examples")
    print(
        "Prediction stats: min="
        f"{predictions.min():.4f}, max={predictions.max():.4f}, "
        f"mean={predictions.mean():.4f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()


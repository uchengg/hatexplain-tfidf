"""Cross-dataset evaluation for the TF-IDF logistic regression classifier.

This is the method I implemented in the five-person HateXplain / Gab Hate
Corpus / Stormfront study. Datasets are not included. Each CSV must have
columns text and label, with label 1 = hate and label 0 = non-hate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.methods.tfidf_logreg import train_and_predict

NAMES = ("hatexplain", "gab_hate_corpus", "stormfront")


def split_one(df: pd.DataFrame, name: str):
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"{name}: removed {removed} duplicate texts ({before} -> {len(df)})")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with hatexplain.csv, gab_hate_corpus.csv, and stormfront.csv",
    )
    args = parser.parse_args()

    frames = {}
    for name in NAMES:
        path = args.data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Place the three processed CSVs there. Do not commit them."
            )
        df = pd.read_csv(path)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames[name] = df

    train_splits = {}
    test_splits = {}
    for name, df in frames.items():
        train_splits[name], test_splits[name] = split_one(df, name)

    for train_name, train_df in train_splits.items():
        for test_name, test_df in test_splits.items():
            pred = train_and_predict(train_df, test_df)
            row = metrics(test_df["label"].to_numpy(), np.asarray(pred))
            print(
                f"{train_name} -> {test_name}: "
                f"accuracy {row['accuracy']:.4f}  macro_f1 {row['macro_f1']:.4f}"
            )


if __name__ == "__main__":
    main()

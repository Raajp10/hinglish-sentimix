"""Shared helpers: data loading, metrics, run output layout."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "results" / "runs"
LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}


def load_split(split: str, normalize: bool = False) -> pd.DataFrame:
    df = pd.read_csv(PROC / f"{split}.csv", dtype={"id": str})
    df["text"] = df["text"].fillna("").astype(str)
    if normalize:
        from src.normalize import load_normalizer
        norm = load_normalizer()
        df["text"] = df["text"].map(norm)
    df["y"] = df["label"].map(L2I)
    return df


def metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=LABELS,
                                output_dict=True, zero_division=0)
    return {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"f1_{l}": rep[l]["f1-score"] for l in LABELS},
        **{f"precision_{l}": rep[l]["precision"] for l in LABELS},
        **{f"recall_{l}": rep[l]["recall"] for l in LABELS},
    }


def safe_name(model_name: str) -> str:
    """'results/tiny_bert' or 'l3cube-pune/hing-bert' -> filesystem-safe tag."""
    return model_name.rstrip("/").split("/")[-1]


def run_dir(model_name: str, seed: int, normalize: bool) -> Path:
    tag = safe_name(model_name) + ("+norm" if normalize else "")
    d = RUNS / tag / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(out: Path, df_test: pd.DataFrame, probs: np.ndarray, val_metrics: dict,
             test_metrics: dict, config: dict) -> None:
    pred = probs.argmax(1)
    pd.DataFrame({
        "id": df_test["id"], "label": df_test["label"], "pred": [LABELS[i] for i in pred],
        **{f"p_{l}": probs[:, i] for i, l in enumerate(LABELS)},
    }).to_csv(out / "test_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(
        {"val": val_metrics, "test": test_metrics, "config": config}, indent=2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

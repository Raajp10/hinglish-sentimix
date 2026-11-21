"""TF-IDF + Logistic Regression baseline (word 1-2 grams + char 2-5 grams).

C is tuned on the validation set; the test set is only scored once.
Usage: python -m src.baseline [--normalize]
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.common import load_split, metrics, run_dir, save_run

C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def featurize(train, *others):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                           max_features=200_000)
    Xtr = hstack([word.fit_transform(train), char.fit_transform(train)]).tocsr()
    return [Xtr] + [hstack([word.transform(o), char.transform(o)]).tocsr() for o in others]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    tr, va, te = (load_split(s, args.normalize) for s in ("train", "val", "test"))
    Xtr, Xva, Xte = featurize(tr["text"], va["text"], te["text"])

    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, tr["y"])
        m = metrics(va["y"], clf.predict(Xva))
        print(f"C={C:<5} val wF1={m['weighted_f1']:.4f}")
        if best is None or m["weighted_f1"] > best[1]["weighted_f1"]:
            best = (C, m, clf)

    C, val_m, clf = best
    probs = clf.predict_proba(Xte)
    test_m = metrics(te["y"], probs.argmax(1))
    out = run_dir("tfidf_lr", 42, args.normalize)
    save_run(out, te, np.asarray(probs), val_m, test_m,
             {"model": "tfidf_lr", "C": C, "normalize": args.normalize, "seed": 42})
    print(f"best C={C}  val wF1={val_m['weighted_f1']:.4f}  TEST wF1={test_m['weighted_f1']:.4f} "
          f"macroF1={test_m['macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()

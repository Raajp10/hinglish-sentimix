"""Parse SentiMix Hinglish CoNLL files -> clean CSVs with Code-Mixing Index.

Outputs data/processed/{train,val,test}.csv with columns:
  id, text, label, n_tokens, n_hin, n_eng, cmi, cmi_bucket, len_bucket

Cleaning:
  * drops @mentions (the '@' token AND the username token after it)
  * drops t.co URLs (tokenised as: https // t . co / <slug>)
  * drops retweet markers (RT), keeps hashtag words without '#'
  * drops trailing ellipsis tokens
  * lower-cases
Leakage fix: the released train file contains ~2.6k tweets that are also in
the validation file. Any train tweet whose cleaned text appears in val or test
is removed, as are duplicate texts inside train.

Usage: python -m src.prepare
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
LABELS = ["negative", "neutral", "positive"]

# CMI buckets (Das & Gambäck 2014 CMI, 0-50 for two languages).
# Fixed thresholds so every model is sliced identically.
CMI_BUCKETS = [(-0.1, 0.0, "mono (0)"), (0.0, 20.0, "low (0-20]"),
               (20.0, 35.0, "medium (20-35]"), (35.0, 100.0, "high (>35)")]
LEN_BUCKETS = [(0, 10, "short (<=10)"), (10, 20, "medium (11-20)"),
               (20, 10_000, "long (>20)")]

URL_PARTS = {"//", "t", ".", "co", "/", ":"}
ELLIPSIS = {"…", "……", "...", "#…", "//…"}


def fix_mojibake(t: str) -> str:
    """Undo UTF-8-read-as-cp1252 damage (e.g. 'â€¦' -> '…', emoji bytes)."""
    if all(ord(c) < 128 for c in t):
        return t
    buf = bytearray()
    for c in t:
        try:
            buf += c.encode("cp1252")
        except UnicodeEncodeError:
            if ord(c) < 256:
                buf.append(ord(c))
            else:
                return t  # genuine non-Latin text (Devanagari, Arabic, ...)
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return t


def parse_conll(path: Path) -> list[dict]:
    tweets, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("meta\t"):
                parts = line.split("\t")
                label = parts[2].strip().lower() if len(parts) > 2 else ""
                cur = {"id": parts[1].strip(), "label": label, "tokens": [], "langs": []}
                tweets.append(cur)
            elif line.strip() and cur is not None:
                parts = line.split("\t")
                tok = fix_mojibake(parts[0])
                lang = parts[1].strip().strip('"') if len(parts) > 1 else "O"
                cur["tokens"].append(tok)
                cur["langs"].append(lang)
    return tweets


def clean_tokens(tokens: list[str], langs: list[str]) -> tuple[list[str], list[str]]:
    out_t, out_l = [], []
    i, n = 0, len(tokens)
    while i < n:
        t, l = tokens[i], langs[i]
        tl = t.lower()
        if t == "@":  # mention: drop '@' and the username
            i += 2
            continue
        if tl in ("http", "https"):  # url: https // t . co / slug
            i += 1
            while i < n and tokens[i].lower() in URL_PARTS:
                prev = tokens[i]
                i += 1
                if prev == "/":  # slug follows the final slash
                    i += 1
                    break
            continue
        if tl == "rt" or t in ELLIPSIS or t == "#":
            i += 1
            continue
        t = t.lstrip("#")
        if t:
            out_t.append(t.lower())
            out_l.append(l)
        i += 1
    return out_t, out_l


def cmi(langs: list[str]) -> tuple[float, int, int]:
    """Code-Mixing Index: 100 * (1 - max_lang / (n - u)); u = lang-independent tokens."""
    n_hin = sum(l == "Hin" for l in langs)
    n_eng = sum(l == "Eng" for l in langs)
    tagged = n_hin + n_eng
    if tagged == 0:
        return 0.0, n_hin, n_eng
    return 100.0 * (1 - max(n_hin, n_eng) / tagged), n_hin, n_eng


def bucket(x: float, buckets) -> str:
    for lo, hi, name in buckets:
        if lo < x <= hi:
            return name
    return buckets[-1][2]


def to_frame(tweets: list[dict]) -> pd.DataFrame:
    rows = []
    for tw in tweets:
        toks, langs = clean_tokens(tw["tokens"], tw["langs"])
        c, nh, ne = cmi(langs)
        rows.append({
            "id": tw["id"], "text": " ".join(toks), "label": tw["label"],
            "n_tokens": len(toks), "n_hin": nh, "n_eng": ne, "cmi": round(c, 2),
            "cmi_bucket": bucket(c, CMI_BUCKETS),
            "len_bucket": bucket(len(toks), LEN_BUCKETS),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = to_frame(parse_conll(RAW / "train.conll"))
    val = to_frame(parse_conll(RAW / "val.conll"))
    test = to_frame(parse_conll(RAW / "test.conll"))

    labels = pd.read_csv(RAW / "test_labels.csv", dtype=str)
    labels.columns = [c.strip() for c in labels.columns]
    lab = dict(zip(labels["Uid"].str.strip(), labels["Sentiment"].str.strip().str.lower()))
    test["label"] = test["id"].map(lab)

    stats = {"raw": {"train": len(train), "val": len(val), "test": len(test)}}

    for name, df in (("train", train), ("val", val), ("test", test)):
        bad = ~df["label"].isin(LABELS)
        stats.setdefault("dropped_bad_label", {})[name] = int(bad.sum())
    train = train[train["label"].isin(LABELS)]
    val = val[val["label"].isin(LABELS)]
    assert test["label"].isin(LABELS).all(), "test labels missing for some ids"

    train = train[train["text"].str.len() > 0]
    held_out = set(val["text"]) | set(test["text"])
    leak = train["text"].isin(held_out)
    stats["train_removed_overlap_with_val_test"] = int(leak.sum())
    train = train[~leak]
    before = len(train)
    train = train.drop_duplicates("text")
    stats["train_removed_internal_duplicates"] = before - len(train)

    for name, df in (("train", train), ("val", val), ("test", test)):
        df.to_csv(OUT / f"{name}.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
        stats.setdefault("final", {})[name] = len(df)
        stats.setdefault("label_dist", {})[name] = df["label"].value_counts().to_dict()
        stats.setdefault("cmi_dist", {})[name] = df["cmi_bucket"].value_counts().to_dict()
        stats.setdefault("len_dist", {})[name] = df["len_bucket"].value_counts().to_dict()
        stats.setdefault("mean_cmi", {})[name] = round(float(df["cmi"].mean()), 2)

    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

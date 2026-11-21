"""Spelling normalization for Romanized Hindi (Goal 4).

Two sources of mappings, merged into one dictionary (variant -> canonical):
  1. SEED: hand-curated frequent variants (always applied).
  2. AUTO: mined from TRAIN only. Hindi-tagged tokens (freq >= MIN_FREQ,
     len >= MIN_LEN) are grouped by a conservative "skeleton" key (ee/oo/w/z
     spelling merges, collapse repeated letters, drop internal schwa 'a',
     keep first+last char). Inside a group, each
     variant maps to the most frequent form if they are also close by string
     similarity (difflib ratio >= MIN_SIM). Mined from train only -> no leakage.

Build:  python -m src.normalize          (writes data/normalization_dict.json)
Use:    from src.normalize import load_normalizer; norm = load_normalizer(); norm(text)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT_PATH = ROOT / "data" / "normalization_dict.json"
MIN_FREQ = 5
MIN_SIM = 0.75

SEED = {
    # negation
    "nhi": "nahi", "nahin": "nahi", "nai": "nahi", "nahee": "nahi", "nhin": "nahi", "nahi.": "nahi",
    # copula / common function words
    "h": "hai", "hain": "hai", "he": "hai", "hy": "hai", "hae": "hai",
    "kya": "kya", "kia": "kya", "kyaa": "kya",
    "kyu": "kyun", "kyon": "kyun", "kyo": "kyun", "q": "kyun",
    "bhi": "bhi", "bi": "bhi",
    "mei": "mein", "me": "mein",
    "mujhe": "mujhe", "muje": "mujhe", "mujhko": "mujhe",
    "tum": "tum", "tm": "tum",
    "hum": "hum", "hm": "hum",
    "aap": "aap", "ap": "aap",
    "kuch": "kuch", "kuchh": "kuch", "kch": "kuch",
    "bahut": "bahut", "bhot": "bahut", "bohot": "bahut", "bahot": "bahut", "bhut": "bahut", "boht": "bahut",
    "accha": "accha", "acha": "accha", "achha": "accha", "achcha": "accha", "acchha": "accha",
    "yaar": "yaar", "yar": "yaar", "yrr": "yaar", "yr": "yaar",
    "matlab": "matlab", "mtlb": "matlab",
    "pyaar": "pyaar", "pyar": "pyaar",
    "sab": "sab", "sb": "sab",
    "kar": "kar", "kr": "kar",
    "raha": "raha", "rha": "raha", "rahi": "rahi", "rhi": "rahi", "rahe": "rahe", "rhe": "rahe",
    "tha": "tha", "thi": "thi", "the": "the",
    "ho": "ho", "hoga": "hoga", "hogaa": "hoga",
    "hua": "hua", "huwa": "hua", "hui": "hui",
    "ke": "ke", "k": "ke",
    "se": "se", "sey": "se",
    "aur": "aur", "or": "aur", "aor": "aur",
    "bhai": "bhai", "bhaii": "bhai", "bhayi": "bhai",
    "abhi": "abhi", "abi": "abhi",
    "phir": "phir", "fir": "phir", "phr": "phir",
    "sahi": "sahi", "shi": "sahi",
    "bilkul": "bilkul", "bilkool": "bilkul",
    "zindagi": "zindagi", "jindagi": "zindagi",
    "kaise": "kaise", "kese": "kaise", "kaisey": "kaise",
    "jaise": "jaise", "jese": "jaise",
    "kyunki": "kyunki", "kyuki": "kyunki", "kyonki": "kyunki",
}
# NOTE: 'he', 'me', 'or', 'the', 'k', 'h', 'q' collide with English. We only
# apply SEED entries whose source word is Hindi-tagged in context when tags are
# available (training-time path); for raw text (demo) ambiguous keys are skipped.
AMBIGUOUS_WITH_ENGLISH = {"he", "me", "or", "the", "h", "k", "q", "hm", "bi", "ap"}

_SPELLING = [("ee", "i"), ("oo", "u"), ("w", "v"), ("z", "j")]
# Auto-mined merges that change meaning (found by manual review). Extend this
# after reviewing data/normalization_dict.json.
BLOCK = {"paas", "banate", "batao", "aapne", "karana", "anna", "mery", "jatay",
         "walay", "hamne", "kren"}
MIN_LEN = 4  # short words are too ambiguous for auto-mining; SEED covers them


def skeleton(w: str) -> str:
    """Conservative key: spelling merges, collapse repeats, drop internal schwa 'a'.

    Two words share a key only if they differ by these edits, e.g.
    bahut/bahot? no; pyaar/pyar yes; karna/krna yes; dekha/dekhe no.
    """
    w = w.lower()
    for a, b in _SPELLING:
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    if len(w) <= 2:
        return w
    return w[0] + w[1:-1].replace("a", "") + w[-1]


def mine(train_csv: Path, raw_train_conll: Path | None = None) -> dict[str, str]:
    """Mine variant->canonical from Hindi-tagged train tokens."""
    from src.prepare import clean_tokens, parse_conll  # local import

    counts: Counter = Counter()
    tweets = parse_conll(raw_train_conll or ROOT / "data" / "raw" / "train.conll")
    import pandas as pd
    keep_ids = set(pd.read_csv(train_csv, dtype={"id": str})["id"])
    for tw in tweets:
        if tw["id"] not in keep_ids:
            continue
        toks, langs = clean_tokens(tw["tokens"], tw["langs"])
        for t, l in zip(toks, langs):
            if l == "Hin" and t.isalpha() and len(t) >= 2:
                counts[t] += 1

    groups: dict[str, list[str]] = defaultdict(list)
    for w, c in counts.items():
        if c >= MIN_FREQ and len(w) >= MIN_LEN:
            groups[skeleton(w)].append(w)

    mapping = {}
    for words in groups.values():
        if len(words) < 2:
            continue
        words.sort(key=lambda w: -counts[w])
        canon = words[0]
        for w in words[1:]:
            if w in BLOCK or canon in BLOCK:
                continue
            if SequenceMatcher(None, w, canon).ratio() >= MIN_SIM:
                mapping[w] = canon
    return mapping


def build() -> dict:
    auto = mine(ROOT / "data" / "processed" / "train.csv")
    merged = dict(auto)
    merged.update({k: v for k, v in SEED.items() if k != v})  # seed wins
    out = {"seed": {k: v for k, v in SEED.items() if k != v}, "auto": auto, "merged": merged,
           "ambiguous_with_english": sorted(AMBIGUOUS_WITH_ENGLISH)}
    DICT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def load_normalizer(path: Path = DICT_PATH, skip_ambiguous: bool = True):
    d = json.loads(Path(path).read_text())["merged"]
    if skip_ambiguous:
        d = {k: v for k, v in d.items() if k not in AMBIGUOUS_WITH_ENGLISH}

    def norm(text: str) -> str:
        return " ".join(d.get(t, t) for t in str(text).split())

    return norm


def changed_fraction(texts, norm) -> float:
    tot = chg = 0
    for t in texts:
        a, b = str(t).split(), norm(t).split()
        tot += len(a)
        chg += sum(x != y for x, y in zip(a, b))
    return chg / max(tot, 1)


if __name__ == "__main__":
    out = build()
    print(f"seed={len(out['seed'])} auto={len(out['auto'])} merged={len(out['merged'])}")
    for k, v in list(out["auto"].items())[:40]:
        print(f"  {k:>14} -> {v}")
    import pandas as pd
    norm = load_normalizer()
    for split in ("train", "val", "test"):
        df = pd.read_csv(ROOT / "data" / "processed" / f"{split}.csv")
        print(f"{split}: {changed_fraction(df['text'], norm):.2%} of tokens changed")

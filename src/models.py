"""Model registry: short name -> HuggingFace id (or local path) + tokenizer kwargs."""
from pathlib import Path

_PRETRAINED = Path(__file__).resolve().parents[1] / "results" / "pretrained"

MODELS = {
    "mbert": {"hf": "bert-base-multilingual-cased", "tok": {}},
    "xlmr": {"hf": "xlm-roberta-base", "tok": {}},
    # IndicBERT v1 is ALBERT + SentencePiece, pre-trained mostly on native scripts.
    # Needs `sentencepiece`; keep_accents per the model card.
    "indicbert": {"hf": "ai4bharat/indic-bert", "tok": {"keep_accents": True, "use_fast": False}},
    # HingBERT: BERT further pre-trained on Romanized Hinglish (L3Cube).
    "hingbert": {"hf": "l3cube-pune/hing-bert", "tok": {}},
    # Optional extras (not in the proposal, cheap to add if time allows):
    "hingroberta": {"hf": "l3cube-pune/hing-roberta", "tok": {}},
    "muril": {"hf": "google/muril-base-cased", "tok": {}},
    # L3Cube's mBERT continued-pretrained on the full HingCorpus: reference for our model.
    "hingmbert": {"hf": "l3cube-pune/hing-mbert", "tok": {}},
    # OUR models (built by src/mlm_pretrain.py, local paths):
    "ourbert": {"hf": str(_PRETRAINED / "ourbert"), "tok": {}},            # mBERT + Hinglish MLM
    "ourbert-tapt": {"hf": str(_PRETRAINED / "ourbert-tapt"), "tok": {}},  # mBERT + MLM on train tweets only
}

CORE = ["mbert", "xlmr", "indicbert", "hingbert"]


def resolve(name: str) -> tuple[str, dict]:
    """Accept a registry key or a raw HF id / local path."""
    if name in MODELS:
        return MODELS[name]["hf"], MODELS[name]["tok"]
    return name, {}

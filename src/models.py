"""Model registry: short name -> HuggingFace id + per-model tokenizer kwargs."""

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
}

CORE = ["mbert", "xlmr", "indicbert", "hingbert"]


def resolve(name: str) -> tuple[str, dict]:
    """Accept a registry key or a raw HF id / local path."""
    if name in MODELS:
        return MODELS[name]["hf"], MODELS[name]["tok"]
    return name, {}

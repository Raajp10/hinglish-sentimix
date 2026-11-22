# Final report outline (fill numbers ONLY from results/tables)

**Title:** Lost in Code-Mixing? Benchmarking Multilingual Transformers on Hinglish Sentiment Analysis
**Team:** Raaj Patel (lead), Amartya Mishra — CS 52570, Group 7

1. **Abstract** (≤150 words): task, 5 models, best wF1 vs 65.4 / 75.0 references, main error finding, normalization result.
2. **Introduction / Motivation**: reuse proposal motivation, shorten. One Hinglish example.
3. **Related work**: SentiMix 2020 overview (Patwa et al.); top system XLM-R + adversarial training (75.0);
   mBERT, XLM-R, IndicBERT (Kakwani et al. 2020), HingBERT (Nayak & Joshi 2022, L3Cube-HingCorpus); CMI (Das & Gambäck 2014).
4. **Data**
   - Raw sizes, cleaning steps (mentions, URLs, RT, emoji mojibake fix, lowercasing).
   - **Leakage finding**: ~2.7k train tweets duplicated in val → removed (data/processed/stats.json).
   - Label distribution table; CMI definition: CMI = 100·(1 − max(n_hin, n_eng)/(n_hin+n_eng)), buckets mono/low/medium/high.
5. **Method**
   - Baseline: TF-IDF word 1-2 + char 2-5, LR, C on val.
   - Transformers: [CLS] classification head, AdamW, linear warmup 10%, max_len 128, bs 32, fp16.
     LR grid {1,2,3,5}e-5 × up to 4 epochs on val (seed 42) → best config × seeds {42,43,44}.
   - Normalization: seed dictionary + train-mined variants (conservative skeleton key + similarity), blocklist after manual check.
   - Metrics: weighted F1 (official), macro F1, per-class F1; paired bootstrap 95% CI for normalization Δ.
6. **Results**: tables/main_results.md + figures/main_results.png. Did we hit 70%? Compare to 65.4 / 75.0. Val vs test gap note.
7. **Error analysis**
   - figures/f1_by_cmi.png, tables/by_cmi.csv — does F1 drop with more mixing? Which model is most robust?
   - figures/f1_by_length.png
   - figures/confusion_matrices.png + tables/neutral_negative.csv — neutral↔negative confusion.
   - Manual review: tables/error_categories.csv + figures/error_categories.png; 3–4 quoted examples.
8. **Normalization experiment**: tables/normalization.csv (Δ, CI). Honest interpretation (subword tokenizers already absorb much variation?).
8b. **Our own Hinglish BERT** (continued MLM pre-training of mBERT; `src/mlm_pretrain.py`, `notebooks/run_ourbert.ipynb`)
   - Setup: base mBERT, MLM 15%, max_len 96, bs 64, lr 5e-5, 3 epochs; corpus sizes and decontamination
     counts from `results/pretrained/<name>/pretrain_info.json`.
   - MLM perplexity before → after on held-out corpus and on SentiMix val text (same file).
   - Fine-tuned test weighted F1 (3 seeds) in one table: mBERT → ourbert-tapt → ourbert → HingMBERT
     (L3Cube, 52.9M tweets) → HingBERT.
   - Interpretation: how much of the mBERT→HingBERT gap does each step close?
     Gap closed = (ours − mBERT) / (HingMBERT − mBERT).
   - Notes: HingLID is CC BY-NC-SA 4.0 (course use OK). 348 SentiMix train tweets share an 8-gram with
     val/test (near-duplicates/retweets); they are excluded from the MLM corpus. Mention as a limitation
     of the classification split too.
9. **Recommendation**: which model for Hinglish in practice, and why (accuracy, robustness to mixing, size).
10. **Limitations & future work**: single dataset (tweets, political domain), label noise, 3 seeds, no adversarial training, no LLM comparison.
11. **Contributions**: Member 1 — fine-tuning, tuning, results table. Member 2 — data pipeline, CMI, baseline, normalization, manual review. Both — report.
12. **References**

## Slides (10–12)
1 Title · 2 Problem + Hinglish example · 3 Dataset + leakage finding · 4 Models & setup · 5 Results table/bar chart ·
6 F1 by CMI · 7 Confusion / neutral-negative · 8 Manual error categories + 2 examples · 9 Normalization result ·
10 Demo (live or screenshots) · 11 Recommendation · 12 Q&A

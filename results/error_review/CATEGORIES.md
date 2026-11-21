Sample: 100 test tweets misclassified by the best transformer (TF-IDF + LR),
stratified by gold label x CMI bucket. Fill `error_category` with ONE of
(use `a; b` only if two clearly apply):

- sarcasm/irony
- non-standard spelling
- heavy code-mixing
- neutral-vs-negative boundary
- needs world/political context
- mixed sentiment
- label noise (gold label looks wrong)
- emoji/hashtag-only signal
- other

Then run: python -m src.analyze --tally results/error_review/to_annotate.csv

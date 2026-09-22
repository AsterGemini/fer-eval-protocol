# Val-error label review

Sample 25 primary-val mistakes so a human can check whether the **official
label** is wrong (FER2013 noise) or the model is.

```bash
# from fer-system/
python -m src.sample_val_errors --checkpoint artifacts/models/efficientnet_b0/checkpoints/best.pt
```

Then open `data/label_review/val_errors/index.html` and mark each official
label Correct / Wrong / Unsure. Download `judgments.json` when done.

Rebuild the val overlay from those judgments:

```bash
python -m src.val_corrections
```

That writes `data/label_review/val_corrections.json`. **The overlay is
disabled** (`data.val_label_corrections: null` since 2026-08-24) and must
stay disabled: remapping "wrong" official labels to the audit-time model
prediction is circular evaluation — the human verdicts never named a
replacement class, and the audit itself showed the model confidently wrong on
8 of 21 decided cases. `judgments.json` and `val_corrections.json` are kept
as audit documentation only. FER val stays frozen with official labels.

The kagglehub ImageFolder is not rewritten. These crops are **not** added to
`extra_train_sources`. RAF-DB val remains the checkpoint monitor.

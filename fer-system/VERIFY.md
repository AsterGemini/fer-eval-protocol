# Verify (public snapshot)

Honest pre-publish checks. **This does not include a full `train.sh` run.**

From `fer-system/` (Python 3.10+ with `requirements.txt` installed):

```bash
python -c "from src.model import FERModel; from data.download import MANIFEST_VERSION; print(MANIFEST_VERSION)"
python -m src.train --help
python -m data.download --help
python -m src.test_training_profile
python -m data.test_no_data_leakage
python -m app.test_ensemble_config
```

Gradio without a local checkpoint should exit with a BYO / train hint:

```bash
python app/app.py   # expect missing-checkpoint message
```

Full training (`scripts/train.sh` / `python -m src.train`) needs Kaggle auth + GPU/CPU time and is **out of scope** for the publish smoke bar.

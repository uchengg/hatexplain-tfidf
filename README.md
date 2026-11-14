# TF-IDF hate-speech classifier

My part of a five-person cross-dataset study on HateXplain, the Gab Hate Corpus, and Stormfront. The classifier is TF-IDF unigrams and bigrams followed by class-balanced logistic regression. The question is whether a detector trained on one dataset still works on the other two.

The three datasets are not included. Other team members wrote the rule-based, embedding, FastText, and transformer runs. Those are not in this folder.

## Run

Put three CSVs in `data/`:

- `hatexplain.csv`
- `gab_hate_corpus.csv`
- `stormfront.csv`

Each file needs `text` and `label`, with `1` for hate and `0` for non-hate.

```powershell
python -m venv .venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_cross_dataset.py
```

`data/` is gitignored.

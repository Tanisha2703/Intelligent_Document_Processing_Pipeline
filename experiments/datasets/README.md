# datasets

Scripts that download the public datasets into `data/real/` (not committed). Sources and licences: [data/README.md](../../data/README.md).

| Script | Downloads |
|---|---|
| [fetch_real.py](fetch_real.py) | SROIE receipts (train and test) |
| [fetch_cord.py](fetch_cord.py) | CORD-v2 receipts |
| [fetch_xfund.py](fetch_xfund.py) | XFUND forms in seven languages |
| [fetch_rvlcdip.py](fetch_rvlcdip.py) | a class-balanced sample of RVL-CDIP |

```bash
python experiments/datasets/fetch_real.py test && python experiments/datasets/fetch_real.py train
python experiments/datasets/fetch_cord.py test train
python experiments/datasets/fetch_xfund.py
python experiments/datasets/fetch_rvlcdip.py
```

# Optional public source datasets

Run `python scripts/fetch_public_sources.py` from the project root to download the Pima Indians Diabetes and UCI Heart Disease reference datasets separately into this folder.

They are intentionally not merged into `data/dataset.csv`: the source feature schemas and label definitions differ from each other and from the uploaded eight-column file. A meaningful harmonisation/multitask design needs to be specified before combining them.

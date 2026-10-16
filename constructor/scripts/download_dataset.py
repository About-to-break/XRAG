
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
out_dir = "data/scifact/raw"

data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(
    data_folder=data_path
).load(split="test")

print(f"Dataset directory: {data_path}")
print(f"Documents: {len(corpus)}")
print(f"Queries: {len(queries)}")
print(f"Queries with relevance labels: {len(qrels)}")

assert len(corpus) > 0
assert len(queries) > 0
assert len(qrels) > 0

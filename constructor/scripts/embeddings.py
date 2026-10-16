import os
import json
from sentence_transformers import SentenceTransformer


def make_embeddings(corpus_path: str, query_path: str):

    def load_records(source_path: str) -> list[dict]:
        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)

        if not source_path.endswith((".json", ".jsonl")):
            raise NotImplementedError(
                "Only json and jsonl are supported"
            )

        with open(source_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_payload(records: list[dict], keys: list[str]) -> list[str]:
        return [
            "\n".join(
                record[key].strip()
                for key in keys
                if isinstance(record.get(key), str)
                and record[key].strip()
            )
            for record in records
        ]

    def attach_embeddings(
        records: list[dict],
        embeddings,
    ) -> list[dict]:
        if len(records) != len(embeddings):
            raise ValueError("Records and embeddings count mismatch")

        result = []
        for record, embedding in zip(records, embeddings):
            result.append({
                **record,
                "embedding": embedding.tolist(),
            })

        return result

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    corpus = load_records(corpus_path)
    queries = load_records(query_path)

    corpus_text = get_payload(corpus, ["title", "text"])
    queries_text = get_payload(queries, ["text"])

    corpus_embeddings = model.encode(
        corpus_text,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        queries_text,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    corpus_vectors = attach_embeddings(corpus, corpus_embeddings)
    query_vectors = attach_embeddings(queries, query_embeddings)

    return corpus_vectors, query_vectors
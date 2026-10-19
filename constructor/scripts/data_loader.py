import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CORPUS_PATH = "data/scifact/raw/scifact/corpus.jsonl"
QUERY_PATH = "data/scifact/raw/scifact/queries.jsonl"
HF_TOKEN = os.getenv("HF_TOKEN")

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
                "embedder_model": EMBEDDING_MODEL,
            })

        return result

    model = SentenceTransformer(EMBEDDING_MODEL, device="cuda", token=HF_TOKEN)
    #records = load_records(QUERY_PATH)

    corpus = load_records(corpus_path)
    queries = load_records(query_path)

    corpus_text = get_payload(corpus, ["title", "text"])
    queries_text = get_payload(queries, ["text"])

    corpus_embeddings = model.encode(
        corpus_text,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cuda",
    )
    query_embeddings = model.encode(
        queries_text,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cuda",
    )

    corpus_vectors = attach_embeddings(corpus, corpus_embeddings)
    query_vectors = attach_embeddings(queries, query_embeddings)

    return corpus_vectors, query_vectors

    # TODO: Add LLM instruction 'Represent this sentence for searching relevant passages: (Scifact docs)'

def load_to_qdrant():
    pass

def main():
    corpus_vectors, query_vectors = make_embeddings(CORPUS_PATH, QUERY_PATH)
    for i in range(3):
        print(f"corpus -> {corpus_vectors[i]}")
        print(f"query -> {query_vectors[i]}")

if __name__ == "__main__":
    main()
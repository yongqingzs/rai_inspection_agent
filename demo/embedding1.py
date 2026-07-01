from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

EMBEDDINGS_URL = "http://10.0.40.149:11452/v1/embeddings"
EMBEDDINGS_MODEL = "qwen3e"
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_patterns: tuple[str, ...]


def dot(va: list[float], vb: list[float]) -> float:
    return sum(a * b for a, b in zip(va, vb))


def normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        EMBEDDINGS_URL,
        json={
            "model": EMBEDDINGS_MODEL,
            "input": texts,
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    return [item["embedding"] for item in payload["data"]]


def load_documents(path: Path) -> list[Document]:
    return [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]


def split_documents(documents: Iterable[Document]) -> list[Document]:
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    chunks: list[Document] = []
    for document in documents:
        markdown_chunks = markdown_splitter.split_text(document.page_content)
        for markdown_chunk in markdown_chunks:
            markdown_chunk.metadata = {
                **document.metadata,
                **markdown_chunk.metadata,
            }
        split_chunks = recursive_splitter.split_documents(markdown_chunks)
        for index, chunk in enumerate(split_chunks):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": index,
            }
        chunks.extend(split_chunks)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def chunk_title(chunk: Document) -> str:
    metadata = chunk.metadata or {}
    for key in ("Header 3", "Header 2", "Header 1"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "<no header>"


def matches_expected(chunk: Document, patterns: tuple[str, ...]) -> bool:
    text = "\n".join(
        [
            chunk.page_content,
            str(chunk.metadata.get("Header 1", "")),
            str(chunk.metadata.get("Header 2", "")),
            str(chunk.metadata.get("Header 3", "")),
            str(chunk.metadata.get("Header 4", "")),
        ]
    )
    return all(re.search(pattern, text) for pattern in patterns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate semantic search quality on man.md")
    parser.add_argument(
        "--doc",
        type=Path,
        default=Path("data/rosbotxl_whoami/documentation/man.md"),
        help="Path to the markdown document to evaluate.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many results to print.")
    parser.add_argument(
        "--pass-k",
        type=int,
        default=8,
        help="The target chunk must appear within this rank to be considered pass.",
    )
    args = parser.parse_args()

    cases = [
        QueryCase(
            query="告诉我机器人的运动性能",
            expected_patterns=(""),
        ),
        QueryCase(
            query="告诉我机器人大小",
            expected_patterns=(""),
        ),
        QueryCase(
            query="温度传感器是哪个",
            expected_patterns=(""),
        ),
    ]

    chunks = split_documents(load_documents(args.doc))
    chunk_texts = [f"search_document: {chunk.page_content}" for chunk in chunks]
    chunk_embeddings = [normalize(vector) for vector in embed(chunk_texts)]

    print(f"document: {args.doc}")
    print(f"chunks: {len(chunks)}")
    print(f"embedding model: {EMBEDDINGS_MODEL}")

    failed = False
    for case in cases:
        query_embedding = normalize(embed([f"search_query: {case.query}"])[0])
        ranked = sorted(
            (
                (dot(query_embedding, chunk_embedding), chunk)
                for chunk_embedding, chunk in zip(chunk_embeddings, chunks, strict=True)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        target_rank = next(
            (
                index
                for index, (_, chunk) in enumerate(ranked, start=1)
                if matches_expected(chunk, case.expected_patterns)
            ),
            None,
        )

        print(f"\nquery: {case.query!r}")
        for index, (score, chunk) in enumerate(ranked[: args.top_k], start=1):
            headers = " > ".join(
                str(chunk.metadata.get(key))
                for key in ("Header 1", "Header 2", "Header 3")
                if chunk.metadata.get(key)
            )
            print(
                f"{index:02d} score={score:.4f} "
                f"chunk={chunk.metadata.get('chunk_index')} "
                f"title={chunk_title(chunk)!r} "
                f"headers={headers!r}"
            )

        if target_rank is None:
            print("  target: not found")
            failed = True
        else:
            print(f"  target_rank: {target_rank}")
            if target_rank > args.pass_k:
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

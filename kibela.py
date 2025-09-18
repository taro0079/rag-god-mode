import os
import sys
import argparse
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings


load_dotenv()

# --- Kibela 設定（環境変数から取得）---
KIBELA_API_URL = os.getenv("KIBELA_API_URL", "https://example.kibe.la/api/v1")
KIBELA_ACCESS_TOKEN = os.getenv("KIBELA_ACCESS_TOKEN")

# --- ベクトルDB設定（redmine.py と同様キー名を流用）---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma")

# --- Azure OpenAI 設定（redmine.py に合わせる）---
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")


def fetch_kibela_notes(after_cursor: Optional[str]) -> Dict[str, Any]:
    if not KIBELA_ACCESS_TOKEN:
        raise RuntimeError("KIBELA_ACCESS_TOKEN is not set.")
    if after_cursor:
        query = f"""
        query {{
          notes(first: 100, after: \"{after_cursor}\") {{
            pageInfo {{
              startCursor
              endCursor
              hasNextPage
            }}
            edges {{
              cursor
              node {{
                title
                createdAt
                updatedAt
                url
                content
              }}
            }}
          }}
        }}
        """
    else:
        query = """
        query {
          notes(first: 100) {
            pageInfo {
              startCursor
              endCursor
              hasNextPage
            }
            edges {
              cursor
              node {
                title
                createdAt
                updatedAt
                url
                content
              }
            }
          }
        }
        """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {KIBELA_ACCESS_TOKEN}",
    }
    try:
        resp = requests.post(
            KIBELA_API_URL,
            headers=headers,
            json={"query": query},
            timeout=60,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception as e:
            # Kibela が HTML を返すなど JSON でない場合のデバッグ出力
            snippet = resp.text[:500]
            raise RuntimeError(
                f"Failed to parse Kibela response as JSON (status={resp.status_code}). Snippet: {snippet}"
            ) from e
    except requests.RequestException as e:
        # 接続系の例外
        raise RuntimeError(f"Kibela request failed: {e}") from e
    if "errors" in data:
        raise RuntimeError(f"Kibela GraphQL error: {data['errors']}")
    return data["data"]["notes"]


def iter_kibela_notes_since(since_iso_utc: str):
    """
    Kibelaのnotesをページングしながら since_iso_utc (ISO8601, UTC) 以降に
    作成または更新された記事をyieldする。
    """
    has_next = True
    cursor: Optional[str] = None

    while has_next:
        notes = fetch_kibela_notes(cursor)
        page_info = notes["pageInfo"]
        has_next = page_info["hasNextPage"]
        cursor = page_info["endCursor"]
        for edge in notes.get("edges", []):
            node = edge.get("node", {})
            # createdAt / updatedAt は ISO8601 を想定
            created_at = node.get("createdAt")
            updated_at = node.get("updatedAt") or created_at
            if updated_at and updated_at >= since_iso_utc:
                yield node


def iter_kibela_notes_all():
    """
    Kibela の notes を全件ページングで取得して yield する。
    """
    has_next = True
    cursor: Optional[str] = None

    while has_next:
        notes = fetch_kibela_notes(cursor)
        page_info = notes["pageInfo"]
        has_next = page_info["hasNextPage"]
        cursor = page_info["endCursor"]
        for edge in notes.get("edges", []):
            node = edge.get("node", {})
            yield node


def build_chunks_and_metadata(
    notes: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, str]]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for n in notes:
        title = (n.get("title") or "").replace("/", "-")
        url = n.get("url") or ""
        created_at = n.get("createdAt") or ""
        updated_at = n.get("updatedAt") or created_at or ""
        content = n.get("content") or ""

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": "kibela",
                    "title": str(title),
                    "url": str(url),
                    "created_at": str(created_at),
                    "updated_at": str(updated_at),
                    "chunk_id": f"{title}-{i}",
                }
            )
    return texts, metadatas


def save_to_vector_db(
    texts: List[str],
    metadatas: List[Dict[str, str]],
    batch_size: int = 64,
    retry_wait_sec: int = 65,
    max_retries: int = 5,
):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_metas = metadatas[start:end]

        attempt = 0
        while True:
            try:
                db.add_texts(texts=batch_texts, metadatas=batch_metas)  # type: ignore[arg-type]
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait = retry_wait_sec * (2 ** (attempt - 1))
                print(
                    f"Embedding batch {start}-{end} failed: {e}. Retry in {wait}s ({attempt}/{max_retries})..."
                )
                time.sleep(wait)

    db.persist()


def main():
    parser = argparse.ArgumentParser(description="Ingest Kibela notes into Chroma")
    parser.add_argument(
        "--full",
        action="store_true",
        help="初回フル取り込み（全記事）。未指定時は前日分のみ",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("KIBELA_EMBED_BATCH_SIZE", "64")),
        help="埋め込みのバッチサイズ (既定: 64)",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=int(os.getenv("KIBELA_RETRY_WAIT_SEC", "65")),
        help="429時の初期待機秒数 (指数バックオフ) (既定: 65)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("KIBELA_MAX_RETRIES", "5")),
        help="429等での最大リトライ回数 (既定: 5)",
    )
    args = parser.parse_args()

    if not KIBELA_ACCESS_TOKEN:
        print("KIBELA_ACCESS_TOKEN が設定されていません", file=sys.stderr)
        sys.exit(1)

    if args.full:
        notes: List[Dict[str, Any]] = list(iter_kibela_notes_all())
    else:
        # 前日 00:00Z 以降を対象
        yesterday_utc = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        notes = list(iter_kibela_notes_since(yesterday_utc))
    if not notes:
        print("No Kibela notes updated yesterday.")
        return

    texts, metadatas = build_chunks_and_metadata(notes)
    if not texts:
        print("No chunks to save.")
        return

    save_to_vector_db(
        texts,
        metadatas,
        batch_size=args.batch_size,
        retry_wait_sec=args.retry_wait,
        max_retries=args.max_retries,
    )
    print(f"{len(texts)} Kibela chunks saved to vector DB at {VECTOR_DB_PATH}.")


if __name__ == "__main__":
    main()

import os
import sys
import time
import argparse
import html
import re
from time import sleep
from typing import Any, Dict, List, Optional, Callable, TypeVar
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from openai import RateLimitError

T = TypeVar("T")


load_dotenv()


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> T:
    """
    レート制限エラーに対して指数バックオフでリトライする関数
    """
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(
                    f"最大リトライ回数 ({max_retries}) に達しました。エラー: {e}",
                    file=sys.stderr,
                )
                raise

            # 指数バックオフで待機時間を計算
            delay = min(base_delay * (2**attempt), max_delay)
            print(
                f"レート制限エラーが発生しました。{delay:.1f}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(delay)
        except Exception as e:
            print(f"予期しないエラーが発生しました: {e}", file=sys.stderr)
            raise

    # この行は到達しないはずですが、型チェッカーのために追加
    raise RuntimeError("予期しないエラーが発生しました")


# --- Relation 設定 ---
RELATION_BASE_URL = os.getenv(
    "RELATION_BASE_URL", "https://customer-precs.relationapp.jp/api/v2/"
)
RELATION_API_KEY = os.getenv("RELATION_API_KEY")
RELATION_MESSAGE_BOX_ID = os.getenv("RELATION_MESSAGE_BOX_ID", "1")

# --- ベクトルDB設定 ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma")

# --- Azure OpenAI 設定 ---
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)


def _headers() -> Dict[str, str]:
    if not RELATION_API_KEY:
        raise RuntimeError("RELATION_API_KEY is not set.")
    return {"Authorization": f"Bearer {RELATION_API_KEY}"}


def _format_ticket_content(content: str) -> str:
    result = html.unescape(content)
    result = re.sub(r"^>.*\n?", "", result, flags=re.MULTILINE)
    result = re.sub(r"<[^>]*>", "", result, flags=re.MULTILINE)
    return result


def _fetch_tickets_page(
    message_box_id: str,
    status: str,
    page: int,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> Dict[str, Any]:
    payloads: Dict[str, Any] = {
        "status_cds": [status],
        "page": page,
    }
    if since:
        payloads["since"] = since
    if until:
        payloads["until"] = until

    url = f"{RELATION_BASE_URL}{message_box_id}/tickets/search"
    resp = requests.post(url, headers=_headers(), json=payloads, timeout=60)
    sleep(0.5)
    resp.raise_for_status()
    return resp.json()


def _fetch_ticket_messages(message_box_id: str, ticket_id: str) -> Dict[str, Any]:
    url = f"{RELATION_BASE_URL}{message_box_id}/tickets/{ticket_id}"
    resp = requests.get(url, headers=_headers(), timeout=60)
    sleep(0.5)
    resp.raise_for_status()
    data = resp.json()
    messages = data.get("messages", [])
    all_message = ""
    for msg in messages:
        all_message += (msg.get("body", "") or "") + "\n"
        all_message += "-----\n"
    return {
        "ticket_id": int(ticket_id),
        "title": data.get("title", ""),
        "content": _format_ticket_content(all_message),
    }


def iter_relation_tickets(
    message_box_id: str,
    status: str,
    since: Optional[str],
    until: Optional[str],
) -> List[Dict[str, Any]]:
    page = 1
    results: List[Dict[str, Any]] = []
    while True:
        page_data = _fetch_tickets_page(message_box_id, status, page, since, until)
        print(page_data)
        tickets = page_data
        if not tickets:
            break
        for t in tickets:
            tid = str(t.get("ticket_id", ""))
            try:
                detail = _fetch_ticket_messages(message_box_id, tid)
                results.append(detail)
            except Exception as e:
                # チケット単位での失敗はスキップし継続
                print(f"ticket {tid} fetch failed: {e}")
                continue
        page += 1
    return results


def build_chunks_and_metadata(
    items: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, str]]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for it in items:
        ticket_id = str(it.get("ticket_id", ""))
        title = str(it.get("title", ""))
        content = str(it.get("content", ""))

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": "relation",
                    "ticket_id": ticket_id,
                    "title": title,
                    "chunk_id": f"{ticket_id}-{i}",
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
    total_batches = (total + batch_size - 1) // batch_size

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_metas = metadatas[start:end]
        batch_num = start // batch_size + 1

        print(
            f"バッチ {batch_num}/{total_batches} を処理中... ({len(batch_texts)} チャンク)"
        )

        def add_batch() -> List[str]:
            return db.add_texts(texts=batch_texts, metadatas=batch_metas)  # type: ignore[arg-type]

        try:
            retry_with_backoff(
                add_batch, max_retries=max_retries, base_delay=retry_wait_sec
            )
            print(f"バッチ {batch_num} 完了")
        except Exception as e:
            print(f"バッチ {batch_num} でエラーが発生しました: {e}", file=sys.stderr)
            raise

    db.persist()


def main():
    parser = argparse.ArgumentParser(description="Ingest Relation tickets into Chroma")
    parser.add_argument(
        "--full",
        action="store_true",
        help="初回フル取り込み（全件）。未指定時は前日分のみ",
    )
    parser.add_argument(
        "--status",
        type=str,
        default=os.getenv("RELATION_STATUS", "closed"),
        help="対象ステータス（例: closed/open）",
    )
    parser.add_argument(
        "--box-id",
        type=str,
        default=RELATION_MESSAGE_BOX_ID,
        help="メッセージボックスID",
    )
    parser.add_argument("--since", type=str, default=None, help="ISO8601で開始日時")
    parser.add_argument("--until", type=str, default=None, help="ISO8601で終了日時")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("RELATION_EMBED_BATCH_SIZE", "64")),
        help="埋め込みのバッチサイズ (既定: 64)",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=int(os.getenv("RELATION_RETRY_WAIT_SEC", "65")),
        help="429時の初期待機秒数 (指数バックオフ) (既定: 65)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("RELATION_MAX_RETRIES", "5")),
        help="429等での最大リトライ回数 (既定: 5)",
    )
    args = parser.parse_args()

    if not RELATION_API_KEY:
        print("RELATION_API_KEY が設定されていません", file=sys.stderr)
        sys.exit(1)

    if args.full:
        since = None
        until = None
    elif args.since:
        since = args.since
        until = None
    else:
        # 前日 00:00Z 以降
        since = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        until = None

    # CLIの明示があれば優先
    if args.since:
        since = args.since
    if args.until:
        until = args.until

    items = iter_relation_tickets(
        message_box_id=args.box_id, status=args.status, since=since, until=until
    )
    if not items:
        print("No Relation tickets to ingest.")
        return

    texts, metadatas = build_chunks_and_metadata(items)
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
    print(f"{len(texts)} Relation chunks saved to vector DB at {VECTOR_DB_PATH}.")


if __name__ == "__main__":
    main()

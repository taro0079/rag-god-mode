import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, TypeVar

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import RateLimitError

T = TypeVar("T")

load_dotenv()  # .env読み込み

# --- helpers ---


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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


def add_texts_with_rate_limit_handling(
    db: Chroma,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 10,
    delay_between_batches: float = 1.0,
) -> None:
    """
    レート制限を考慮してテキストをバッチでベクトルDBに追加する関数
    """
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(
            f"バッチ {batch_num}/{total_batches} を処理中... ({len(batch_texts)} チャンク)"
        )

        def add_batch() -> List[str]:
            return db.add_texts(texts=batch_texts, metadatas=batch_metadatas)

        try:
            retry_with_backoff(add_batch)
            print(f"バッチ {batch_num} 完了")

            # バッチ間の待機（レート制限を避けるため）
            if batch_num < total_batches:
                time.sleep(delay_between_batches)

        except Exception as e:
            print(f"バッチ {batch_num} でエラーが発生しました: {e}", file=sys.stderr)
            raise


# --- Redmine設定 ---
REDMINE_URL = os.getenv("REDMINE_URL", "https://redmine.example.com")
API_KEY = os.getenv("REDMINE_API_KEY")

# --- ベクトルDB設定 ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_redmine")

# --- Azure OpenAI 設定 ---
# 必須: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
# 埋め込みは「デプロイ名（azure_deployment）」で指定するのが確実
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# --- CLI defaults ---
DEFAULT_STATUS_ID = os.getenv("REDMINE_STATUS_ID", "*")
DEFAULT_LOOKBACK_DAYS = int(os.getenv("REDMINE_LOOKBACK_DAYS", "1"))
DEFAULT_PER_PAGE = int(os.getenv("REDMINE_PER_PAGE", "100"))
DEFAULT_CHUNK_SIZE = int(os.getenv("REDMINE_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("REDMINE_CHUNK_OVERLAP", "200"))
DEFAULT_TIMEOUT = int(os.getenv("REDMINE_TIMEOUT", "60"))
DEFAULT_BATCH_SIZE = int(os.getenv("REDMINE_BATCH_SIZE", "10"))
DEFAULT_BATCH_DELAY = float(os.getenv("REDMINE_BATCH_DELAY", "1.0"))
DEFAULT_INCLUDE_DESCRIPTION = _env_flag("REDMINE_INCLUDE_DESCRIPTION", True)
DEFAULT_INCLUDE_JOURNALS = _env_flag("REDMINE_INCLUDE_JOURNALS", True)
DEFAULT_INCLUDE_CUSTOM_FIELDS = _env_flag("REDMINE_INCLUDE_CUSTOM_FIELDS", False)
DEFAULT_INCLUDE_METADATA = _env_flag("REDMINE_INCLUDE_METADATA", True)


def fetch_updated_issues(
    base_url: str,
    api_key: str,
    *,
    status_id: Optional[str],
    updated_on: Optional[str],
    include_journals: bool,
    project_id: Optional[str],
    tracker_id: Optional[str],
    assigned_to_id: Optional[str],
    category_id: Optional[str],
    per_page: int,
    limit: Optional[int],
    max_pages: Optional[int],
    extra_params: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    """Fetch issues from Redmine using pagination and flexible filters."""

    url = base_url.rstrip("/") + "/issues.json"
    issues: List[Dict[str, Any]] = []
    offset = 0
    page_count = 0

    while True:
        params: Dict[str, Any] = {
            "key": api_key,
        }
        if status_id:
            params["status_id"] = status_id
        if include_journals:
            params["include"] = "journals"
        if updated_on:
            params["updated_on"] = updated_on
        if project_id:
            params["project_id"] = project_id
        if tracker_id:
            params["tracker_id"] = tracker_id
        if assigned_to_id:
            params["assigned_to_id"] = assigned_to_id
        if category_id:
            params["category_id"] = category_id
        if per_page > 0:
            params["limit"] = per_page
        if offset > 0:
            params["offset"] = offset
        if extra_params:
            params.update(extra_params)

        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        page_issues = payload.get("issues", [])
        total_count = payload.get("total_count")

        if not page_issues:
            break

        issues.extend(page_issues)
        if limit and len(issues) >= limit:
            issues = issues[:limit]
            break

        offset += len(page_issues)
        page_count += 1

        if total_count is not None and offset >= total_count:
            break
        if per_page > 0 and len(page_issues) < per_page:
            break
        if max_pages and page_count >= max_pages:
            break

    return issues


def issue_to_text(
    issue: Dict[str, Any],
    *,
    include_description: bool,
    include_journals: bool,
    include_custom_fields: bool,
    include_metadata: bool,
) -> str:
    lines: List[str] = []
    lines.append(f"[#{issue['id']}] {issue.get('subject', '')}")

    if include_metadata:
        status = issue.get("status", {}).get("name")
        tracker = issue.get("tracker", {}).get("name")
        priority = issue.get("priority", {}).get("name")
        assignee = issue.get("assigned_to", {}).get("name")
        updated_on = issue.get("updated_on")
        metadata_parts = []
        if status:
            metadata_parts.append(f"ステータス: {status}")
        if tracker:
            metadata_parts.append(f"トラッカー: {tracker}")
        if priority:
            metadata_parts.append(f"優先度: {priority}")
        if assignee:
            metadata_parts.append(f"担当者: {assignee}")
        if updated_on:
            metadata_parts.append(f"更新日: {updated_on}")
        if metadata_parts:
            lines.append(" | ".join(metadata_parts))

    if include_description and issue.get("description"):
        lines.append(f"説明: {issue['description']}")

    if include_custom_fields:
        for field in issue.get("custom_fields", []):
            name = field.get("name")
            value = field.get("value")
            if name and value:
                lines.append(f"カスタム項目({name}): {value}")

    if include_journals:
        for journal in issue.get("journals", []):
            notes = journal.get("notes")
            if notes:
                author = journal.get("user", {}).get("name", "unknown")
                created = journal.get("created_on", "")
                lines.append(f"コメント({author} @ {created}): {notes}")

    return "\n".join(lines)


def build_updated_on_filter(
    *,
    explicit: Optional[str],
    since: Optional[str],
    until: Optional[str],
    days: Optional[int],
) -> Optional[str]:
    if explicit:
        return explicit
    if since and until:
        return f"><{since}|{until}"
    if since:
        return f">={since}"
    if until:
        return f"<={until}"
    if days is not None:
        base_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        return f">={base_date}"
    return None


def parse_extra_params(params: Optional[List[str]]) -> Dict[str, str]:
    extra: Dict[str, str] = {}
    if not params:
        return extra
    for raw in params:
        if "=" not in raw:
            raise ValueError(f"Invalid extra param '{raw}'. Use key=value format.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid extra param '{raw}'. Key is empty.")
        extra[key] = value.strip()
    return extra


def resolve_metadata_date(
    *, metadata_date: Optional[str], since: Optional[str], days: Optional[int]
) -> str:
    if metadata_date:
        return metadata_date
    if since:
        return since
    if days is not None:
        return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Redmine issues into a Chroma vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--base-url", default=REDMINE_URL, help="Redmine base URL")
    parser.add_argument("--api-key", default=API_KEY, help="Redmine API key")
    parser.add_argument(
        "--status",
        dest="status_id",
        default=DEFAULT_STATUS_ID,
        help="Redmine status_id filter (e.g. * / open / closed)",
    )
    parser.add_argument("--project-id", help="Filter by project_id")
    parser.add_argument("--tracker-id", help="Filter by tracker_id")
    parser.add_argument("--assigned-to-id", help="Filter by assigned_to_id")
    parser.add_argument("--category-id", help="Filter by category_id")

    parser.add_argument("--since", help="YYYY-MM-DD 形式で updated_on の開始日")
    parser.add_argument("--until", help="YYYY-MM-DD 形式で updated_on の終了日")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="今日から何日前までを対象にするか (since/until/updated-filter が未指定の場合)",
    )
    parser.add_argument(
        "--no-date-filter",
        action="store_true",
        help="updated_on フィルターを適用しない",
    )
    parser.add_argument(
        "--updated-filter",
        help="Redmine updated_on パラメータを直接指定 (例: '><2024-01-01|2024-01-31')",
    )
    parser.add_argument(
        "--metadata-date",
        help="メタデータ date に設定する値 (省略時は since/days から推定)",
    )

    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="Redmine API の limit (1〜100)。0 で未指定",
    )
    parser.add_argument("--limit", type=int, help="取得する最大件数")
    parser.add_argument(
        "--max-pages",
        type=int,
        help="ページングの最大回数 (制限しない場合は未指定)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Redmine API 呼び出しのタイムアウト秒",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="追加のクエリパラメータ (key=value)。複数指定可",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="テキスト分割時のチャンクサイズ",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="テキスト分割時のオーバーラップ",
    )
    parser.add_argument(
        "--vector-path",
        default=VECTOR_DB_PATH,
        help="Chroma の永続化ディレクトリ",
    )
    parser.add_argument(
        "--embedding-deployment",
        default=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        help="Azure OpenAI の埋め込みデプロイ名",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="レート制限回避のためのバッチサイズ",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=DEFAULT_BATCH_DELAY,
        help="バッチ間の待機時間（秒）",
    )

    parser.add_argument(
        "--include-description",
        dest="include_description",
        action="store_true",
        default=DEFAULT_INCLUDE_DESCRIPTION,
        help="課題の説明を含める",
    )
    parser.add_argument(
        "--no-description",
        dest="include_description",
        action="store_false",
        help="課題の説明を除外",
    )
    parser.add_argument(
        "--include-journals",
        dest="include_journals",
        action="store_true",
        default=DEFAULT_INCLUDE_JOURNALS,
        help="ジャーナル(コメント)を含める",
    )
    parser.add_argument(
        "--no-journals",
        dest="include_journals",
        action="store_false",
        help="ジャーナルを除外",
    )
    parser.add_argument(
        "--include-custom-fields",
        dest="include_custom_fields",
        action="store_true",
        default=DEFAULT_INCLUDE_CUSTOM_FIELDS,
        help="カスタムフィールドを本文に含める",
    )
    parser.add_argument(
        "--no-custom-fields",
        dest="include_custom_fields",
        action="store_false",
        help="カスタムフィールドを除外",
    )
    parser.add_argument(
        "--include-metadata",
        dest="include_metadata",
        action="store_true",
        default=DEFAULT_INCLUDE_METADATA,
        help="ステータスや担当者などのメタ情報を本文に含める",
    )
    parser.add_argument(
        "--no-metadata",
        dest="include_metadata",
        action="store_false",
        help="メタ情報を除外",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ベクトルDBへ保存せず件数のみ表示",
    )
    parser.add_argument("--verbose", action="store_true", help="詳細ログを出力")

    args = parser.parse_args()

    if not args.api_key:
        print("REDMINE_API_KEY が設定されていません", file=sys.stderr)
        sys.exit(1)

    try:
        extra_params = parse_extra_params(args.param)
    except ValueError as err:
        parser.error(str(err))

    days = None if args.no_date_filter else args.days
    updated_on_filter = build_updated_on_filter(
        explicit=args.updated_filter,
        since=args.since,
        until=args.until,
        days=days,
    )
    metadata_date = resolve_metadata_date(
        metadata_date=args.metadata_date,
        since=args.since,
        days=days,
    )

    if args.verbose:
        print(
            f"Fetching Redmine issues from {args.base_url} with updated_on={updated_on_filter} "
            f"status={args.status_id} project={args.project_id} limit={args.limit}"
        )

    issues = fetch_updated_issues(
        base_url=args.base_url,
        api_key=args.api_key,
        status_id=args.status_id,
        updated_on=updated_on_filter,
        include_journals=args.include_journals,
        project_id=args.project_id,
        tracker_id=args.tracker_id,
        assigned_to_id=args.assigned_to_id,
        category_id=args.category_id,
        per_page=args.per_page,
        limit=args.limit,
        max_pages=args.max_pages,
        extra_params=extra_params,
        timeout=args.timeout,
    )

    if not issues:
        print("対象のRedmineチケットが見つかりませんでした。")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    base_url_clean = args.base_url.rstrip("/")

    for issue in issues:
        issue_id = issue.get("id")
        if issue_id is None:
            continue

        doc = issue_to_text(
            issue,
            include_description=args.include_description,
            include_journals=args.include_journals,
            include_custom_fields=args.include_custom_fields,
            include_metadata=args.include_metadata,
        )
        if not doc.strip():
            continue

        chunks = splitter.split_text(doc)
        updated_on = issue.get("updated_on")
        subject = issue.get("subject")
        project_name = issue.get("project", {}).get("name")

        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            meta: Dict[str, Any] = {
                "source": "redmine",
                "issue_id": issue_id,
                "chunk_id": f"{issue_id}-{idx}",
                "url": f"{base_url_clean}/issues/{issue_id}",
                "date": metadata_date,
            }
            if updated_on:
                meta["updated_on"] = updated_on
            if subject:
                meta["subject"] = subject
            if project_name:
                meta["project"] = project_name
            metadatas.append(meta)

    if not texts:
        print("チャンクが生成されませんでした。フィルタ設定を見直してください。")
        return

    if args.verbose:
        print(f"Prepared {len(texts)} chunks from {len(issues)} issues.")

    if args.dry_run:
        print(f"[DRY RUN] {len(texts)} chunks would be saved to {args.vector_path}.")
        return

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=args.embedding_deployment,
    )

    db = Chroma(persist_directory=args.vector_path, embedding_function=embeddings)

    # レート制限を考慮したバッチ処理でテキストを追加
    print(f"レート制限を考慮して {len(texts)} チャンクをバッチ処理します...")
    add_texts_with_rate_limit_handling(
        db,
        texts,
        metadatas,
        batch_size=args.batch_size,
        delay_between_batches=args.batch_delay,
    )
    db.persist()

    print(f"{len(texts)} Redmine chunks saved to vector DB at {args.vector_path}.")


if __name__ == "__main__":
    main()

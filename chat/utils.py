from typing import List, Dict
from langchain_core.documents import Document

def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    RAGの「コンテキスト」欄へ流し込む文字列を生成。
    メタデータに issue_id, updated_on, url などが入っている想定。
    """
    lines = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        issue_id = md.get("issue_id") or md.get("id") or "unknown"
        updated = md.get("updated_on") or md.get("date") or ""
        url = md.get("url") or ""
        lines.append(f"- [#{issue_id}] (updated: {updated}) {url}\n{d.page_content}")
    return "\n\n".join(lines)

def extract_citations(docs: List[Document]) -> List[Dict]:
    """
    APIレスポンスに含めるための引用情報を抽出。
    """
    cites = []
    for d in docs:
        md = d.metadata or {}
        cites.append({
            "issue_id": md.get("issue_id") or md.get("id"),
            "chunk_id": md.get("chunk_id"),
            "updated_on": md.get("updated_on") or md.get("date"),
            "url": md.get("url"),
            "score": md.get("score"),  # スコアを付ける場合
        })
    # 重複issue_idはユニーク化してもよい
    return cites

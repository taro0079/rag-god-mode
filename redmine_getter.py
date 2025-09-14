import os
import requests
from datetime import datetime, timedelta
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 環境変数などから設定
REDMINE_URL = os.getenv("REDMINE_URL", "https://redmine.example.com")
API_KEY = os.getenv("REDMINE_API_KEY")
VECTOR_DB_PATH = "./chroma_redmine"

# 1. 更新日の指定（前日分を対象）
yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

def fetch_updated_issues():
    url = f"{REDMINE_URL}/issues.json"
    params = {
        "key": API_KEY,
        "updated_on": f">={yesterday}",
        "status_id": "*",       # 全ステータス対象
        "include": "journals"   # コメント（ジャーナル）も含める
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()["issues"]

def issue_to_text(issue):
    lines = []
    lines.append(f"[#{issue['id']}] {issue['subject']}")
    if issue.get("description"):
        lines.append(f"説明: {issue['description']}")
    # コメント部分
    if "journals" in issue:
        for j in issue["journals"]:
            notes = j.get("notes")
            if notes:
                author = j.get("user", {}).get("name", "unknown")
                created = j.get("created_on", "")
                lines.append(f"コメント({author} @ {created}): {notes}")
    return "\n".join(lines)

def main():
    issues = fetch_updated_issues()
    docs = [issue_to_text(issue) for issue in issues]

    # 2. チャンク分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = []
    for doc in docs:
        texts.extend(splitter.split_text(doc))

    # 3. ベクトルDBに保存
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    metadatas = [{"source": "redmine", "date": yesterday}] * len(texts)
    db.add_texts(texts, metadatas=metadatas)

    db.persist()
    print(f"{len(texts)} chunks saved to vector DB.")

if __name__ == "__main__":
    main()

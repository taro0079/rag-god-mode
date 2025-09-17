import os
import requests
from datetime import datetime, timedelta
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()  # .env読み込み

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
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
)

# 1. 更新日の指定（前日分を対象; 00:00Z〜の単純フィルタ）
yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

def fetch_updated_issues():
    """
    Redmineから「昨日以降に更新されたチケット」をjournals込みで取得。
    必要に応じてlimit/offsetでページングを追加してください。
    """
    url = f"{REDMINE_URL}/issues.json"
    params = {
        "key": API_KEY,
        "updated_on": f">={yesterday}",
        "status_id": "*",       # 全ステータス対象
        "include": "journals"   # コメント（ジャーナル）も含める
        # "limit": 100, "offset": 0, など
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("issues", [])

def issue_to_text(issue: dict) -> str:
    lines = []
    lines.append(f"[#{issue['id']}] {issue.get('subject','')}")
    if issue.get("description"):
        lines.append(f"説明: {issue['description']}")
    # コメント部分（journals.notes）
    for j in issue.get("journals", []):
        notes = j.get("notes")
        if notes:
            author = j.get("user", {}).get("name", "unknown")
            created = j.get("created_on", "")
            lines.append(f"コメント({author} @ {created}): {notes}")
    return "\n".join(lines)

def main():
    issues = fetch_updated_issues()
    if not issues:
        print("No updated issues.")
        return

    # 2. チャンク分割
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts, metadatas = [], []

    for issue in issues:
        doc = issue_to_text(issue)
        chunks = splitter.split_text(doc)

        issue_id = issue.get("id")
        updated_on = issue.get("updated_on")
        url = f"{REDMINE_URL}/issues/{issue_id}"

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": "redmine",
                "issue_id": issue_id,
                "updated_on": updated_on,
                "url": url,
                "chunk_id": f"{issue_id}-{i}",
                "date": yesterday,
            })

    # 3. Azure OpenAI Embeddings でベクトル化→Chromaに保存
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        # openai_api_key / azure_endpoint は環境変数から自動読込
    )

    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    db.add_texts(texts=texts, metadatas=metadatas)
    db.persist()

    print(f"{len(texts)} chunks saved to vector DB at {VECTOR_DB_PATH}.")

if __name__ == "__main__":
    main()

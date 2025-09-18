起動方法
uv sync  # uvを使わない場合は: pip install -r requirements.txt
uv run uvicorn app:app --reload --port 8001
# もしくは: uvicorn app:app --reload --port 8000

# 確認方法
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RPSTXの定期便スキップはどこで設定しますか？",
    "history": []
  }' | jq

# ディレクトリ構成
redmine-rag-chat/
├─ app.py                       # FastAPIエントリ（/chat）
├─ rag_chain.py                 # RAGチェーン定義（LangChain/LCEL）
├─ utils.py                     # ドキュメント整形・引用抽出など
├─ prompts/
│   └─ chat/ja/support_rag.v1.yaml
├─ requirements.txt
└─ .env.example

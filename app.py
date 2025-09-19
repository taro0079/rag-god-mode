import os
from typing import List, Literal, Any, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_chain import support_chain
from markdown_it import MarkdownIt

load_dotenv()  # .env読み込み

# LangSmithトレーシングの設定確認
if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
    print("LangSmithトレーシングが有効です")
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("警告: LANGCHAIN_API_KEYが設定されていません")

app = FastAPI(title="Redmine RAG Chat API")

# テンプレート設定
templates = Jinja2Templates(directory="templates")

allowed_origins = os.getenv("CORS_ALLOW_ORIGINS")
origins = (
    [o.strip() for o in allowed_origins.split(",") if o.strip()]
    if allowed_origins
    else [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://rpst-n8n-test.precs.info",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    used_docs: int


_chain = support_chain()

_markdown_renderer = (
    MarkdownIt("commonmark", {"html": False, "linkify": True})
    .enable("table")
    .enable("strikethrough")
)


def render_markdown_to_html(markdown_text: str) -> str:
    """Render markdown text to HTML for assistant responses."""
    return _markdown_renderer.render(markdown_text or "").strip()


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """チャットページのメインエンドポイント"""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # LangChainのMessagesPlaceholder用フォーマットに合わせる
    history = [(m.role, m.content) for m in req.history]
    result: Dict[str, Any] = _chain.invoke(
        {"question": req.question, "history": history}
    )
    return ChatResponse(**result)


@app.post("/chat-htmx", response_class=HTMLResponse)
async def chat_htmx(request: Request):
    """HTMX用のチャットエンドポイント"""
    form_data = await request.json()

    question = form_data.get("question", "")
    history_str = form_data.get("history", "[]")

    if not question:
        return HTMLResponse("<div class='error'>質問を入力してください。</div>")

    try:
        import json

        history_data = json.loads(history_str)
        history = [(m["role"], m["content"]) for m in history_data]
    except:
        history = []

    # RAGチェーンを実行
    result: Dict[str, Any] = _chain.invoke({"question": question, "history": history})

    # 新しいメッセージを履歴に追加
    new_history = history + [("user", question), ("assistant", result["answer"])]

    return templates.TemplateResponse(
        "chat_response.html",
        {
            "request": request,
            "answer": result["answer"],
            "answer_html": render_markdown_to_html(result["answer"]),
            "citations": result["citations"],
            "used_docs": result["used_docs"],
            "history": new_history,
        },
    )

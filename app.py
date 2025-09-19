import os
from typing import List, Literal, Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_chain import support_chain

load_dotenv()  # .env読み込み
app = FastAPI(title="Redmine RAG Chat API")

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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # LangChainのMessagesPlaceholder用フォーマットに合わせる
    history = [(m.role, m.content) for m in req.history]
    result: Dict[str, Any] = _chain.invoke(
        {"question": req.question, "history": history}
    )
    return ChatResponse(**result)

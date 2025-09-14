import os
from typing import List, Literal, Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_chain import support_chain

load_dotenv()  # .env読み込み
app = FastAPI(title="Redmine RAG Chat API")

class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    citations: list
    used_docs: int

_chain = support_chain()

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # LangChainのMessagesPlaceholder用フォーマットに合わせる
    history = [(m.role, m.content) for m in req.history]
    result: Dict[str, Any] = _chain.invoke({
        "question": req.question,
        "history": history
    })
    return ChatResponse(**result)

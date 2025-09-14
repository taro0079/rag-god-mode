import os
from typing import Dict, Any, List, Literal, Optional
import yaml
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Azure利用時は:
# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils import format_docs_for_prompt, extract_citations

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_redmine")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))

def load_prompt_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_llm():
    """
    OpenAI or Azure OpenAI を状況に応じて初期化。
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        # Azure利用時の例（モデル名はデプロイ名）
        # return AzureChatOpenAI(
        #     azure_deployment=MODEL_NAME,
        #     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        #     temperature=0,
        # )
        raise NotImplementedError("Azure使用時は上のコメントを有効化してください。")
    else:
        return ChatOpenAI(model=MODEL_NAME, temperature=0)

def build_embeddings():
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        # return AzureOpenAIEmbeddings(model="text-embedding-3-small")
        raise NotImplementedError("Azure使用時は上のコメントを有効化してください。")
    else:
        return OpenAIEmbeddings(model="text-embedding-3-small")

def build_retriever():
    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=build_embeddings()
    )
    return db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

def support_chain():
    """
    LCELで RAG チェーンを組む。
    入力: {"question": str, "history": [messages?]}
    出力: {"answer": str, "citations": [...], "used_docs": int}
    """
    # 1) プロンプト読込
    spec = load_prompt_yaml("prompts/chat/ja/support_rag.v1.yaml")
    system_msg = next(m for m in spec["messages"] if m["role"] == "system")["content"]
    user_msg = next(m for m in spec["messages"] if m["role"] == "user")["content"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder("history"),          # クライアント側の履歴を渡せる
        ("user", user_msg),
    ])

    llm = build_llm()
    retriever = build_retriever()

    # 2) 検索器：question から docs を取得
    def _retrieve(inputs: Dict[str, Any]) -> List[Document]:
        q = inputs["question"]
        docs = retriever.get_relevant_documents(q)
        # スコアをmetadataに埋め込む例（Chroma標準は返さないので省略）
        # for d in docs: d.metadata["score"] = ...
        return docs

    retrieve = RunnableLambda(_retrieve)

    # 3) コンテキスト整形
    def _make_context(docs: List[Document]) -> str:
        return format_docs_for_prompt(docs)

    to_context = RunnableLambda(_make_context)

    # 4) citations抽出
    to_citations = RunnableLambda(extract_citations)

    # 5) 並列に docs と入力を束ねる
    # {"question": ..., "history": ...} => {"question": ..., "context": "...formatted...", "citations": [...], "docs": [...]}
    with_context = RunnableParallel(
        context = retrieve | to_context,
        citations = retrieve | to_citations,
        docs = retrieve
    ) | RunnablePassthrough.assign(
        question = lambda x: x["question"]
    )

    # 6) 生成
    generate = (
        prompt
        | llm
        | StrOutputParser()
    )

    # 7) 最終出力整形
    def _finalize(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # inputs: {"question", "history"} → with_context で {"context","citations","docs"} が付与される
        gen = generate.invoke({
            "history": inputs.get("history", []),
            "question": inputs["question"],
            "context": inputs["context"],
        })
        return {
            "answer": gen,
            "citations": inputs["citations"],
            "used_docs": len(inputs["docs"]),
        }

    finalize = RunnableLambda(_finalize)

    # 8) 全体パイプ
    return with_context | finalize

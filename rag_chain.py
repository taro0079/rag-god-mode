import os
import time
from typing import Dict, Any, List, Callable, TypeVar
import yaml
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import RateLimitError

# Azure利用時は:
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils import format_docs_for_prompt, extract_citations

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """
    レート制限エラーに対して指数バックオフでリトライする関数
    """
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"最大リトライ回数 ({max_retries}) に達しました。エラー: {e}")
                raise

            # 指数バックオフで待機時間を計算
            delay = min(base_delay * (2**attempt), max_delay)
            print(
                f"レート制限エラーが発生しました。{delay:.1f}秒待機してリトライします... (試行 {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
        except Exception as e:
            print(f"予期しないエラーが発生しました: {e}")
            raise

    # この行は到達しないはずですが、型チェッカーのために追加
    raise RuntimeError("予期しないエラーが発生しました")


def load_prompt_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm():
    """
    OpenAI or Azure OpenAI を状況に応じて初期化。
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        # Azure利用時の例（モデル名はデプロイ名）
        return AzureChatOpenAI(
            azure_deployment=MODEL_NAME,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            temperature=0,
        )
        raise NotImplementedError("Azure使用時は上のコメントを有効化してください。")
    else:
        return ChatOpenAI(model=MODEL_NAME, temperature=0)


def build_embeddings():
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )
        # raise NotImplementedError("Azure使用時は上のコメントを有効化してください。")
    else:
        return OpenAIEmbeddings(model="text-embedding-3-large")


def build_retriever():
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=build_embeddings())
    return db.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def support_chain():
    """
    LCELで RAG チェーンを組む。
    入力: {"question": str, "history": [messages?]}
    出力: {"answer": str, "citations": [...], "used_docs": int}
    """
    # 1) プロンプト読込
    spec = load_prompt_yaml("prompts/prompt.yaml")
    system_msg = next(m for m in spec["messages"] if m["role"] == "system")["content"]
    user_msg = next(m for m in spec["messages"] if m["role"] == "user")["content"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            MessagesPlaceholder("history"),  # クライアント側の履歴を渡せる
            ("user", user_msg),
        ]
    )

    llm = build_llm()
    retriever = build_retriever()

    # 2) 検索器：question から docs を取得（スコアを metadata に埋め込む）
    def _retrieve(inputs: Dict[str, Any]) -> List[Document]:
        q = inputs["question"]

        def search_with_scores() -> List[Document]:
            # retriever の裏の vectorstore (Chroma) を直接呼んでスコアを取得
            vectorstore = getattr(retriever, "vectorstore", None)
            if vectorstore is not None and hasattr(
                vectorstore, "similarity_search_with_relevance_scores"
            ):
                results = vectorstore.similarity_search_with_relevance_scores(
                    q, k=RETRIEVAL_K
                )
                docs_only: List[Document] = []
                for doc, score in results:
                    try:
                        doc.metadata["score"] = float(score)
                    except Exception:
                        pass
                    docs_only.append(doc)
                return docs_only
            else:
                # フォールバック: 通常のRetriever（スコアなし）
                return retriever.get_relevant_documents(q)

        try:
            return retry_with_backoff(search_with_scores)
        except Exception as e:
            print(f"検索でエラーが発生しました: {e}")
            # 最終フォールバック: 通常のRetriever（スコアなし）
            try:
                return retriever.get_relevant_documents(q)
            except Exception as fallback_error:
                print(f"フォールバック検索でもエラーが発生しました: {fallback_error}")
                return []

    retrieve = RunnableLambda(_retrieve)

    def _to_context(docs: List[Document]) -> str:
        return format_docs_for_prompt(docs)

    def _to_citations(docs: List[Document]) -> List[Dict[str, Any]]:
        return extract_citations(docs)

    # 5) 並列に docs と入力を束ねる
    # {"question": ..., "history": ...} => {"question": ..., "context": "...formatted...", "citations": [...], "docs": [...]}
    # with_context = RunnableParallel(
    #     context = retrieve | to_context,
    #     citations = retrieve | to_citations,
    #     docs = retrieve
    # ) | RunnablePassthrough.assign(
    #     question = lambda x: x["question"]
    # )
    with_context = RunnablePassthrough.assign(
        docs=retrieve,
    ).assign(
        context=lambda x: _to_context(x["docs"]),
        citations=lambda x: _to_citations(x["docs"]),
    )

    # 6) 生成
    generate = prompt | llm | StrOutputParser()

    # 7) 最終出力整形
    def _finalize(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # inputs: {"question", "history"} → with_context で {"context","citations","docs"} が付与される

        def generate_response() -> str:
            return generate.invoke(
                {
                    "history": inputs.get("history", []),
                    "question": inputs["question"],
                    "context": inputs["context"],
                }
            )

        try:
            gen = retry_with_backoff(generate_response)
        except Exception as e:
            print(f"LLM生成でエラーが発生しました: {e}")
            gen = "申し訳ございませんが、現在レート制限により回答を生成できません。しばらく時間をおいてから再度お試しください。"

        return {
            "answer": gen,
            "citations": inputs.get("citations", []),
            "used_docs": len(inputs.get("docs", [])),
        }

    finalize = RunnableLambda(_finalize)

    # 8) 全体パイプ
    return with_context | finalize

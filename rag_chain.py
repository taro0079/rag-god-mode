import os
import time
from typing import Dict, Any, List, Callable, TypeVar, Iterator
import yaml
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import RateLimitError
from langsmith import Client

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

# LangSmith設定
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-god-mode")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# LangSmithクライアントの初期化
langsmith_client = None
if LANGCHAIN_TRACING_V2 and os.getenv("LANGCHAIN_API_KEY"):
    try:
        langsmith_client = Client()
        print(
            f"LangSmithトレーシングが有効になりました。プロジェクト: {LANGCHAIN_PROJECT}"
        )
    except Exception as e:
        print(f"LangSmithクライアントの初期化に失敗しました: {e}")
        langsmith_client = None

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


class SupportChain:
    """Wrapper providing both invoke and streaming helpers for the RAG chain."""

    def __init__(self, *, chain, with_context, generate):
        self._chain = chain
        self._with_context = with_context
        self._generate = generate

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Proxy to the underlying chain's invoke method."""
        return self._chain.invoke(inputs)

    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Async variant mirroring LangChain's ainvoke."""
        return await self._chain.ainvoke(inputs)

    def stream_answer(self, inputs: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Yield streaming events for the assistant's answer along with metadata."""

        context_inputs = self._with_context.invoke(inputs)

        llm_inputs = {
            "history": context_inputs.get("history", inputs.get("history", [])),
            "question": context_inputs["question"],
            "context": context_inputs.get("context", ""),
        }

        answer_buffer: List[str] = []

        try:
            for chunk in self._generate.stream(llm_inputs):
                answer_buffer.append(chunk)
                yield {"type": "token", "content": chunk}
        except Exception as error:  # pragma: no cover - lifecycle delegation
            yield {
                "type": "error",
                "message": str(error),
            }
            return

        answer_text = "".join(answer_buffer)

        yield {
            "type": "done",
            "answer": answer_text,
            "citations": context_inputs.get("citations", []),
            "used_docs": len(context_inputs.get("docs", [])),
        }

    def __getattr__(self, item: str):
        return getattr(self._chain, item)


def support_chain():
    """
    LCELで RAG チェーンを組む。
    入力: {"question": str, "history": [messages?]}
    出力: {"answer": str, "citations": [...], "used_docs": int}
    """
    # LangSmithトレーシングの設定
    if LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
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
    base_chain = with_context | finalize

    # LangSmith用のメタデータを追加
    if LANGCHAIN_TRACING_V2:
        base_chain = base_chain.with_config(
            configurable={
                "metadata": {
                    "project": LANGCHAIN_PROJECT,
                    "model": MODEL_NAME,
                    "retrieval_k": RETRIEVAL_K,
                    "max_tokens": MAX_TOKENS,
                }
            }
        )

    return SupportChain(chain=base_chain, with_context=with_context, generate=generate)

#!/usr/bin/env python3
"""
LangSmithトレーシングのテストスクリプト
"""
import os
from dotenv import load_dotenv
from rag_chain import support_chain


def test_langsmith_tracing():
    """LangSmithトレーシングをテストする"""
    load_dotenv()

    # 環境変数の確認
    print("=== LangSmith設定確認 ===")
    print(f"LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2', 'false')}")
    print(
        f"LANGCHAIN_API_KEY: {'設定済み' if os.getenv('LANGCHAIN_API_KEY') else '未設定'}"
    )
    print(f"LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT', 'rag-god-mode')}")
    print()

    # RAGチェーンの初期化
    print("=== RAGチェーン初期化 ===")
    chain = support_chain()
    print("RAGチェーンが正常に初期化されました")
    print()

    # テスト質問
    test_question = "Redmineの使い方について教えてください"
    print(f"=== テスト実行: {test_question} ===")

    try:
        result = chain.invoke({"question": test_question, "history": []})

        print("✅ テストが正常に完了しました")
        print(f"回答: {result['answer'][:100]}...")
        print(f"引用数: {len(result['citations'])}")
        print(f"使用ドキュメント数: {result['used_docs']}")

        if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
            print("\n✅ LangSmithトレーシングが有効です")
            print("LangSmithダッシュボードでトレースを確認してください:")
            print("https://smith.langchain.com/")
        else:
            print("\n⚠️  LangSmithトレーシングが無効です")
            print(
                "トレーシングを有効にするには、.envファイルでLANGCHAIN_TRACING_V2=trueを設定してください"
            )

    except Exception as e:
        print(f"❌ テスト中にエラーが発生しました: {e}")
        return False

    return True


if __name__ == "__main__":
    test_langsmith_tracing()

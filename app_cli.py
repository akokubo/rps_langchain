import random
from langchain_openai import ChatOpenAI  # OpenAIのLLMを使用
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os

# === じゃんけんゲーム ===
# LangChainを使った対話型のじゃんけんゲーム。
# アシスタント（LLM）がユーザーの選択と履歴を考慮しながら手を決める。

# ===============================
# グローバル変数（履歴）
# ===============================
history = []  # じゃんけんの履歴を保存するリスト
choices = ["グー", "チョキ", "パー"]  # じゃんけんの選択肢

# ===============================
# ゲームの状態（State）を定義
# ===============================
class GameState(TypedDict):
    user_choice: str         # ユーザーの選択
    assistant_choice: str    # アシスタントの選択
    result: str              # 勝敗の結果
    history: list            # 過去の履歴

# ===============================
# ツール（履歴の取得・追加）
# ===============================

@tool
def get_history():
    """過去のじゃんけんの履歴を取得する"""
    return history

@tool
def add_to_history(user_choice: str, assistant_choice: str, result: str):
    """今回のじゃんけんの結果を履歴に追加する"""
    history.append({
        "ラウンド": len(history) + 1,
        "あなた": user_choice,
        "アシスタント": assistant_choice,
        "結果": result
    })
    return "履歴に追加しました"

# ===============================
# LLM（ChatOpenAI）設定
# ===============================
# ChatOpenAIモデルのインスタンスを作成
llm = ChatOpenAI(
    model_name="gemma3",
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="ollama",
    temperature=0.6,  # 応答の創造性を調整
)

# プロンプトテンプレート（アシスタントの手を決めるための指示）
prompt = ChatPromptTemplate.from_template(
    """
    あなたはじゃんけんの対戦相手です。
    過去の履歴: {history}
    選択肢: グー、チョキ、パー
    ユーザーの傾向を考慮し、勝つ可能性を高める手を「[選択: 手]」の形式で返してください。
    """
)

# ===============================
# ノード（ワークフローの各処理）
# ===============================

def choose_assistant_move(state: GameState):
    """
    アシスタントの手を決定するノード。
    LLMの応答を基に手を決めるが、不正な出力があればランダム選択する。
    """
    # 過去の履歴を取得
    current_history = get_history.invoke({})
    
    # LLMにプロンプトを送信し、じゃんけんの手を決める
    # response = llm.invoke(prompt.format(user_choice=state["user_choice"], history=str(current_history)))
    response = llm.invoke(prompt.format(user_choice=state["user_choice"], history=str(current_history)))
    
    # LLMの応答から手を抽出
    # assistant_choice = response.strip().replace("[選択: ", "").replace("]", "")
    assistant_choice = response.content.strip().replace("[選択: ", "").replace("]", "")


    # 万が一、LLMの応答が不適切だった場合、ランダムに手を選ぶ
    if assistant_choice not in choices:
        assistant_choice = random.choice(choices)
    
    # 決定した手を状態に保存
    state["assistant_choice"] = assistant_choice
    return state

def determine_result_and_update(state: GameState):
    """
    勝敗を判定し、履歴を更新するノード。
    ルールに従って結果を決定し、履歴に追加する。
    """
    user_choice = state["user_choice"]
    assistant_choice = state["assistant_choice"]

    # じゃんけんの勝敗判定
    if user_choice == assistant_choice:
        result = "あいこです"
    elif (user_choice == "グー" and assistant_choice == "チョキ") or \
         (user_choice == "チョキ" and assistant_choice == "パー") or \
         (user_choice == "パー" and assistant_choice == "グー"):
        result = "あなたの勝ちです"
    else:
        result = "あなたの負けです"
    
    # 状態を更新
    state["result"] = result

    # 履歴に追加
    add_to_history.invoke({
        "user_choice": user_choice,
        "assistant_choice": assistant_choice,
        "result": result
    })

    return state

# ===============================
# ワークフロー（StateGraph）の設定
# ===============================

# ゲームの状態遷移を管理するグラフを作成
workflow = StateGraph(GameState)

# 各ノード（処理ステップ）を追加
workflow.add_node("choose_move", choose_assistant_move)
workflow.add_node("determine_result", determine_result_and_update)

# ノード間の遷移を定義
workflow.add_edge("choose_move", "determine_result")  # アシスタントの手を決めたら、勝敗判定へ
workflow.add_edge("determine_result", END)  # 勝敗判定が終わったら終了

# 開始地点を設定
workflow.set_entry_point("choose_move")

# グラフをコンパイルして実行可能にする
app = workflow.compile()

# ===============================
# メインゲームループ
# ===============================

print("=== じゃんけんゲーム ===")

while True:
    # ユーザーの入力を取得
    user_choice = input("グー、チョキ、パーのいずれかを選んでください（やめる場合は「やめる」と入力）: ")

    # ゲーム終了条件
    if user_choice == "やめる":
        break

    # 無効な入力を防ぐ
    if user_choice not in choices:
        print("無効な入力です。グー、チョキ、パーのいずれかを選んでください。")
        continue

    # グラフ（ワークフロー）を実行
    state = app.invoke({"user_choice": user_choice, "history": history})

    # 結果を表示
    print(f"あなた: {user_choice}")
    print(f"アシスタント: {state['assistant_choice']}")
    print(state["result"])

# ===============================
# ゲーム履歴の表示
# ===============================

print("\n=== ゲーム履歴 ===")
if history:
    for record in history:
        print(f"ラウンド {record['ラウンド']}: あなた: {record['あなた']} vs アシスタント: {record['アシスタント']} - {record['結果']}")
else:
    print("履歴はありません。")

print("ゲームを終了しました。")

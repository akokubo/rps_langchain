import streamlit as st
import random
from langchain_openai import ChatOpenAI  # OpenAIのLLMを使用
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os

# ===============================
# グローバル変数（履歴）
# ===============================
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
# LLM（ChatOpenAI）設定
# ===============================
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
    """アシスタントの手を決定するノード"""
    current_history = state["history"]

    # LLMにプロンプトを送信し、じゃんけんの手を決める
    response = llm.invoke(prompt.format(user_choice=state["user_choice"], history=str(current_history)))
    
    # LLMの応答から手を抽出
    assistant_choice = response.content.strip().replace("[選択: ", "").replace("]", "")

    # 万が一、LLMの応答が不適切だった場合、ランダムに手を選ぶ
    if assistant_choice not in choices:
        assistant_choice = random.choice(choices)
    
    # 決定した手を状態に保存
    state["assistant_choice"] = assistant_choice
    return state

def determine_result_and_update(state: GameState):
    """勝敗を判定し、履歴を更新するノード"""
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
    state["history"].append({
        "ラウンド": len(state["history"]) + 1,
        "あなた": user_choice,
        "アシスタント": assistant_choice,
        "結果": result
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
# Streamlit UI
# ===============================

st.title("🤖 じゃんけんゲーム")

# セッション状態を初期化
if "history" not in st.session_state:
    st.session_state.history = []
if "user_choice" not in st.session_state:
    st.session_state.user_choice = None
if "assistant_choice" not in st.session_state:
    st.session_state.assistant_choice = None
if "result" not in st.session_state:
    st.session_state.result = None

# じゃんけんの選択肢をボタンで表示
st.write("グー、チョキ、パーのいずれかを選んでください:")
col1, col2, col3 = st.columns(3)
if col1.button("✊ グー"):
    st.session_state.user_choice = "グー"
if col2.button("✌️ チョキ"):
    st.session_state.user_choice = "チョキ"
if col3.button("🖐 パー"):
    st.session_state.user_choice = "パー"

# ユーザーの選択があった場合、ゲームを実行
if st.session_state.user_choice:
    state = {
        "user_choice": st.session_state.user_choice,
        "assistant_choice": "",
        "result": "",
        "history": st.session_state.history,
    }

    # ワークフローを実行
    state = app.invoke(state)

    # 結果をセッション状態に保存
    st.session_state.assistant_choice = state["assistant_choice"]
    st.session_state.result = state["result"]
    st.session_state.history = state["history"]

    # 結果を表示
    st.subheader("🎮 結果")
    st.write(f"あなた: **{st.session_state.user_choice}**")
    st.write(f"アシスタント: **{st.session_state.assistant_choice}**")
    st.write(f"📝 {st.session_state.result}")

# 過去の履歴を表示
st.subheader("📜 過去の対戦履歴")
if st.session_state.history:
    for record in reversed(st.session_state.history):  # 最新のものを上に表示
        st.write(f"ラウンド {record['ラウンド']}: あなた: {record['あなた']} vs アシスタント: {record['アシスタント']} - **{record['結果']}**")
else:
    st.write("まだ対戦履歴はありません。")
